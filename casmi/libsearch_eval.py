"""
Validation of casmi.libsearch on a simulated class-1 problem.

Query sets
  np    : the 250 enveda-np-examples molecules (same instrument + pipeline as the hidden
          test set).  Library = every other ingest_lib; reported with and without enveda-180.
  gnps  : 300 random gnps structures that also occur in at least one other library.
          Queries = their gnps spectra, library = all non-gnps libraries.

Stages (run one or several):
  python -m casmi.libsearch_eval recall      mass-window recall ceiling + candidate counts
  python -m casmi.libsearch_eval grid        similarity-configuration sweep
  python -m casmi.libsearch_eval calib       score calibration + class-2/3 top-score distribution
  python -m casmi.libsearch_eval robust      the chosen config on the gnps query set
  python -m casmi.libsearch_eval timing      library build + search wall clock
"""

import os
import sys
import time
import json
import itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from casmi.libsearch import (SpectralLibrary, SearchConfig, estimate_molecule_mass,
                             mass_window, DEFAULT_MASS_PPM)
from casmi.metric import mrr_at_25

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, 'train.parquet')
STRUCTS = os.path.join(ROOT, 'work', 'structures.parquet')
W = os.path.join(ROOT, 'work', 'libsearch')
N_PROC = 5

# The configuration picked by the sweep (config C4 -- see work/libsearch/REPORT.md).
CHOSEN = SearchConfig(tol_da=0.02, power=0.5, mz_power=0.0, min_rel=0.005, max_peaks=60,
                      drop_precursor=1.5, modified=False, min_matches=2,
                      same_mode_only=True, agg='mean_top2')


# ─────────────────────────────────────────────────────────────────────
# Query sets
# ─────────────────────────────────────────────────────────────────────

def _to_queries(df, key='inchikey14'):
    out = {}
    for k, g in df.groupby(key, sort=True):
        out[k] = [{'mzs': r.ms2_mzs, 'intensities': r.ms2_normalized_intensities,
                   'precursor_mz': float(r.precursor_mz), 'adduct': r.adduct,
                   'ionization_mode': r.ionization_mode} for r in g.itertuples(index=False)]
    return out


def load_set(name):
    """-> (queries, truth_smiles, masses, library, structures)"""
    S = pd.read_parquet(STRUCTS, columns=['ik14', 'smiles', 'mass', 'n_spectra', 'mkey'])
    smi = dict(zip(S.ik14, S.smiles))
    if name == 'np':
        df = pd.read_pickle(os.path.join(W, 'np_queries.pkl'))
        lib_path = os.path.join(W, 'lib_np.npz')
    else:
        df = pd.read_pickle(os.path.join(W, 'gnps_queries.pkl'))
        lib_path = os.path.join(W, 'lib_gnps.npz')
    q = _to_queries(df)
    truth = {k: smi[k] for k in q}
    masses = {k: estimate_molecule_mass([s['precursor_mz'] for s in v],
                                        [s['adduct'] for s in v],
                                        [s['ionization_mode'] for s in v]) for k, v in q.items()}
    lib = SpectralLibrary.load(lib_path)
    return q, truth, masses, lib, S


# ─────────────────────────────────────────────────────────────────────
# Parallel search
# ─────────────────────────────────────────────────────────────────────

_G = {}


def _init(lib, queries, masses, ppm, blind, xlib=None):
    _G['lib'], _G['q'], _G['m'], _G['ppm'], _G['blind'] = lib, queries, masses, ppm, blind
    _G['xlib'] = xlib


def _one(mid):
    m = _G['m'][mid]
    if not np.isfinite(m):
        return mid, pd.DataFrame(columns=['ik14', 'smiles', 'score', 'n_lib_spectra'])
    lo, hi = mass_window(m, _G['ppm'])
    ex_ik = _G['blind'].get(mid, ()) if isinstance(_G['blind'], dict) else ((mid,) if _G['blind'] else ())
    ex_lib = _G.get('xlib', {}).get(mid, ()) if isinstance(_G.get('xlib'), dict) else ()
    return mid, _G['lib'].search(_G['q'][mid], lo, hi, exclude_libs=ex_lib, exclude_ik14=ex_ik)


def run_search(lib, queries, masses, cfg, ppm=DEFAULT_MASS_PPM, blind=False, n_proc=N_PROC, xlib=None):
    lib.set_config(cfg)
    _init(lib, queries, masses, ppm, blind, xlib)
    ids = list(queries)
    t0 = time.time()
    if n_proc > 1:
        with Pool(n_proc) as p:
            res = dict(p.map(_one, ids, chunksize=8))
    else:
        res = dict(_one(i) for i in ids)
    return res, time.time() - t0


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────

def score(res, truth, order='score'):
    """order: 'score' (as returned), 'random', 'n_spectra'"""
    rng = np.random.default_rng(0)
    guesses = {}
    for mid, df in res.items():
        if len(df) == 0:
            guesses[mid] = []
            continue
        if order == 'score':
            g = df.smiles.tolist()
        elif order == 'random':
            idx = rng.permutation(len(df))
            g = df.smiles.values[idx].tolist()
        else:
            idx = np.argsort(-df.n_lib_spectra.values, kind='stable')
            g = df.smiles.values[idx].tolist()
        guesses[mid] = g[:25]
    mrr, ranks = mrr_at_25(guesses, truth)
    r = np.array(list(ranks.values()))
    return dict(mrr=mrr, top1=float((r == 1).mean()), top5=float(((r > 0) & (r <= 5)).mean()),
                top25=float((r > 0).mean()), n=len(r), ranks=ranks)


def fmt(name, s, extra=''):
    se = s.get('se')
    se_s = ' +/-%.3f' % se if se else ''
    return ('%-46s MRR %.4f%s  top1 %.3f  top5 %.3f  top25 %.3f  n=%d %s'
            % (name, s['mrr'], se_s, s['top1'], s['top5'], s['top25'], s['n'], extra))


def bootstrap_se(ranks, n_boot=2000, seed=0):
    rr = np.array([1.0 / r if r > 0 else 0.0 for r in ranks.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rr), size=(n_boot, len(rr)))
    return float(rr[idx].mean(axis=1).std())


# ─────────────────────────────────────────────────────────────────────
# Stage: recall ceiling
# ─────────────────────────────────────────────────────────────────────

def stage_recall():
    S = pd.read_parquet(STRUCTS, columns=['ik14', 'smiles', 'mass', 'n_spectra'])
    sm = np.sort(S.mass.values)
    out = []
    for name in ('np', 'gnps'):
        df = pd.read_pickle(os.path.join(W, '%s_queries.pkl' % name))
        q = _to_queries(df)
        true_mass = dict(zip(S.ik14, S.mass))
        est = np.array([estimate_molecule_mass([s['precursor_mz'] for s in v],
                                               [s['adduct'] for s in v],
                                               [s['ionization_mode'] for s in v]) for v in q.values()])
        tru = np.array([true_mass[k] for k in q])
        ppm_err = (est - tru) / tru * 1e6
        print('\n== %s (%d molecules) ==' % (name, len(q)))
        print('calibrated mass error ppm: median %.2f  |err| p50 %.2f p90 %.2f p99 %.2f max %.2f'
              % (np.median(ppm_err), *np.percentile(np.abs(ppm_err), [50, 90, 99, 100])))
        for ppm in (5, 10, 15):
            lo, hi = est * (1 - ppm * 1e-6), est * (1 + ppm * 1e-6)
            nc = np.searchsorted(sm, hi, 'right') - np.searchsorted(sm, lo, 'left')
            rec = float(((tru >= lo) & (tru <= hi)).mean())
            print('  +/-%2d ppm: true-structure-in-window %.4f | candidates median %.0f p90 %.0f max %d'
                  % (ppm, rec, np.median(nc), np.percentile(nc, 90), nc.max()))
            out.append(dict(set=name, ppm=ppm, recall=rec, med=np.median(nc),
                            p90=np.percentile(nc, 90), max=int(nc.max())))
    return out


# ─────────────────────────────────────────────────────────────────────
# Stage: configuration sweep
# ─────────────────────────────────────────────────────────────────────

def _sweep(lib, queries, truth, masses, configs, tag, se=False, xlib=None):
    rows = []
    for name, cfg in configs:
        res, dt = run_search(lib, queries, masses, cfg, xlib=xlib)
        s = score(res, truth)
        if se:
            s['se'] = bootstrap_se(s['ranks'])
        ncand = np.array([len(d) for d in res.values()])
        print(fmt('%s | %s' % (tag, name), s, '(%.1fs, cand med %d)' % (dt, np.median(ncand))))
        rows.append(dict(tag=tag, config=name, seconds=dt, **{k: v for k, v in s.items() if k != 'ranks'}))
    return rows


def stage_grid(quick=False):
    q, truth, masses, lib, S = load_set('np')
    lib180 = lib.drop_libs(['enveda-180'])
    print('library: %d spectra / %d structures; without enveda-180: %d spectra'
          % (len(lib.spec_prec), len(lib.ik14), len(lib180.spec_prec)))

    base = SearchConfig()
    configs = []
    # 1. tolerance
    for t in (0.005, 0.01, 0.02, 0.05):
        configs.append(('tol=%.3f' % t, base.copy(tol_da=t)))
    # 2. intensity transform
    for p in (1.0, 0.5, 0.3, 0.4):
        configs.append(('power=%.1f' % p, base.copy(power=p)))
    # 3. m/z weighting
    for mp in (0.0, 0.5, 1.0):
        configs.append(('mz_power=%.1f' % mp, base.copy(mz_power=mp)))
    # 4. intensity floor
    for f in (0.0, 0.005, 0.01, 0.02):
        configs.append(('min_rel=%.3f' % f, base.copy(min_rel=f)))
    # 5. peak cap
    for n in (30, 60, 100, 200):
        configs.append(('max_peaks=%d' % n, base.copy(max_peaks=n)))
    # 6. precursor region
    for d in (0.0, 1.5, 5.0):
        configs.append(('drop_precursor=%.1f' % d, base.copy(drop_precursor=d)))
    # 7. matched-peak floor
    for m in (1, 2, 3, 4):
        configs.append(('min_matches=%d' % m, base.copy(min_matches=m)))
    # 8. aggregation
    for a in ('max', 'mean_qmax', 'mean_qmax_all', 'mean_top2', 'mean_top3'):
        configs.append(('agg=%s' % a, base.copy(agg=a)))
    # 9. ion mode / adduct handling
    configs.append(('cross_mode', base.copy(same_mode_only=False)))
    configs.append(('adduct_penalty=0.2', base.copy(same_adduct_bonus=0.2)))
    # 10. modified cosine
    configs.append(('modified', base.copy(modified=True)))

    if quick:
        configs = configs[:6]

    rows = _sweep(lib, q, truth, masses, configs, 'A(all-but-np)')

    # baselines
    res, _ = run_search(lib, q, masses, base)
    for o in ('random', 'n_spectra'):
        s = score(res, truth, order=o)
        print(fmt('A(all-but-np) | BASELINE %s' % o, s))
        rows.append(dict(tag='A(all-but-np)', config='BASELINE ' + o, seconds=0.0,
                         **{k: v for k, v in s.items() if k != 'ranks'}))

    pd.DataFrame(rows).to_csv(os.path.join(W, 'grid_stage1.csv'), index=False)

    # refinement around the best of stage 1
    best = max(rows, key=lambda r: r['mrr'] if not r['config'].startswith('BASELINE') else -1)
    print('\nstage-1 best: %s (%.4f)' % (best['config'], best['mrr']))
    return rows


COMBOS = [
    ('default              ', SearchConfig()),
    ('C1 t.01 p.5 f.005 n60', SearchConfig(tol_da=0.01, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax')),
    ('C2 t.02 p.5 f.005 n60', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax')),
    ('C3 = C2 + agg all    ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax_all')),
    ('C4 = C2 + agg top2   ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C5 = C2 + agg max    ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='max')),
    ('C6 = C2 + mz_power 1 ', SearchConfig(tol_da=0.02, power=0.5, mz_power=1.0, min_rel=0.005,
                                           max_peaks=60, drop_precursor=1.5, min_matches=2,
                                           agg='mean_qmax')),
    ('C7 = C2 + n30        ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=30,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax')),
    ('C8 = C2 + modified   ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax',
                                           modified=True)),
    ('C9 = C2 + keep prec  ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=0.0, min_matches=2, agg='mean_qmax')),
    ('C10= C2 + no transform', SearchConfig(tol_da=0.02, power=1.0, min_rel=0.005, max_peaks=60,
                                            drop_precursor=1.5, min_matches=2, agg='mean_qmax')),
    ('C11= C2 + tol .005   ', SearchConfig(tol_da=0.005, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax')),
    ('C12= C2 + tol .05    ', SearchConfig(tol_da=0.05, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_qmax')),
]


def stage_grid2(combos=None):
    """Combine the individually-best knobs and confirm on all three query/library variants."""
    combos = combos or COMBOS
    rows = []
    q, truth, masses, lib, S = load_set('np')
    lib180 = lib.drop_libs(['enveda-180'])
    rows += _sweep(lib, q, truth, masses, combos, 'A(all-but-np)', se=True)
    print()
    rows += _sweep(lib180, q, truth, masses, combos, 'B(no enveda-180)', se=True)
    print()
    for tag, L in (('A(all-but-np)', lib), ('B(no enveda-180)', lib180)):
        res, _ = run_search(L, q, masses, CHOSEN)
        for o in ('random', 'n_spectra'):
            s = score(res, truth, order=o)
            s['se'] = bootstrap_se(s['ranks'])
            print(fmt('%s | BASELINE %s' % (tag, o), s))
            rows.append(dict(tag=tag, config='BASELINE ' + o, seconds=0.0,
                             **{k: v for k, v in s.items() if k != 'ranks'}))
    print()
    del lib, lib180
    qg, truthg, massesg, libg, _ = load_set('gnps')
    rows += _sweep(libg, qg, truthg, massesg, combos, 'C(gnps)', se=True)
    res, _ = run_search(libg, qg, massesg, CHOSEN)
    for o in ('random', 'n_spectra'):
        s = score(res, truthg, order=o)
        s['se'] = bootstrap_se(s['ranks'])
        print(fmt('C(gnps) | BASELINE %s' % o, s))
        rows.append(dict(tag='C(gnps)', config='BASELINE ' + o, seconds=0.0,
                         **{k: v for k, v in s.items() if k != 'ranks'}))
    pd.DataFrame(rows).to_csv(os.path.join(W, 'grid_stage2.csv'), index=False)
    return rows


FINE = [
    ('C4  (tol .02, top2)  ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + tol .01+10ppm   ', SearchConfig(tol_da=0.01, tol_ppm=10.0, power=0.5, min_rel=0.005,
                                           max_peaks=60, drop_precursor=1.5, min_matches=2,
                                           agg='mean_top2')),
    ('C4 + min_matches 1   ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=1, agg='mean_top2')),
    ('C4 + min_matches 3   ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=3, agg='mean_top2')),
    ('C4 + agg top3        ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top3')),
    ('C4 + floor 0.0       ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.0, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + floor 0.01      ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.01, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + floor 0.02      ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.02, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + n30             ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=30,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + n100            ', SearchConfig(tol_da=0.02, power=0.5, min_rel=0.005, max_peaks=100,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + power 0.4       ', SearchConfig(tol_da=0.02, power=0.4, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
    ('C4 + power 0.6       ', SearchConfig(tol_da=0.02, power=0.6, min_rel=0.005, max_peaks=60,
                                           drop_precursor=1.5, min_matches=2, agg='mean_top2')),
]


def add_noise(queries, seed=0, extra=1.4, max_rel=0.03):
    """Make the validation queries look as noisy as the hidden test set.

    The enveda-np-examples rows in train.parquet have been through Enveda's cleaning
    (median 141 peaks, 61% of peaks below 1% of base) while the hidden test spectra have
    not (median 230 peaks, 83% below 1%).  Random low-intensity peaks are added so the two
    distributions roughly match; this checks the intensity floor / peak cap are doing their job.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for mid, specs in queries.items():
        new = []
        for s in specs:
            mz = np.asarray(s['mzs'], float)
            it = np.asarray(s['intensities'], float)
            hi = float(s['precursor_mz']) + 2.0 if np.isfinite(s['precursor_mz']) else mz.max()
            k = int(len(mz) * extra)
            nmz = rng.uniform(50.0, max(hi, 60.0), size=k)
            nit = it.max() * np.exp(rng.uniform(np.log(1e-4), np.log(max_rel), size=k))
            d = dict(s)
            d['mzs'] = np.concatenate([mz, nmz])
            d['intensities'] = np.concatenate([it, nit])
            new.append(d)
        out[mid] = new
    return out


def stage_fine():
    q, truth, masses, lib, S = load_set('np')
    rows = _sweep(lib, q, truth, masses, FINE, 'A(all-but-np)', se=True)
    print()
    qn = add_noise(q)
    rows += _sweep(lib, qn, truth, masses, FINE, 'A-noisy', se=True)
    print()
    del lib
    qg, truthg, massesg, libg, _ = load_set('gnps')
    rows += _sweep(libg, qg, truthg, massesg, FINE, 'C(gnps)', se=True)
    pd.DataFrame(rows).to_csv(os.path.join(W, 'grid_fine.csv'), index=False)
    return rows


# ─────────────────────────────────────────────────────────────────────
# Stage: calibration
# ─────────────────────────────────────────────────────────────────────

def _bucket_table(vals, correct, edges, label):
    lines = ['%-14s %7s %7s %9s' % (label, 'n', 'P(hit)', 'share')]
    vals = np.asarray(vals, float)
    correct = np.asarray(correct, bool)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (vals >= a) & (vals < b if b < edges[-1] else vals <= b)
        n = int(m.sum())
        p = float(correct[m].mean()) if n else float('nan')
        lines.append('%-14s %7d %7s %9.3f' % ('[%.2f,%.2f)' % (a, b), n,
                                              ('%.3f' % p) if n else '  -  ', n / max(len(vals), 1)))
    return '\n'.join(lines)


def _top_stats(res, truth, mkeys):
    top_score, margin, correct = [], [], []
    for mid, df in res.items():
        if len(df) == 0:
            continue
        tk = mkeys[truth[mid]]
        s0 = float(df.score.values[0])
        s1 = float(df.score.values[1]) if len(df) > 1 else 0.0
        top_score.append(s0)
        margin.append(s0 - s1)
        correct.append(mkeys[df.smiles.values[0]] == tk)
    return np.array(top_score), np.array(margin), np.array(correct)


def stage_calib(which=('np', 'gnps')):
    out = {}
    for setname in which:
        q, truth, masses, lib, S = load_set(setname)
        # class-2/3 simulation: hide every structure that shares the truth's *metric key*
        # (tautomers / stereoisomers with a different ik14 would otherwise still be scored
        # as correct, so hiding only the exact ik14 overstates the false-alarm rate).
        by_mkey = S.groupby('mkey').ik14.apply(list).to_dict()
        mk_of = dict(zip(S.ik14, S.mkey))
        blind_map = {mid: by_mkey.get(mk_of.get(mid), [mid]) for mid in q}
        variants = ([('A(all-but-np)', lib), ('B(no enveda-180)', lib.drop_libs(['enveda-180']))]
                    if setname == 'np' else [('C(gnps)', lib)])
        for tag, L in variants:
            res, _ = run_search(L, q, masses, CHOSEN)
            resb, _ = run_search(L, q, masses, CHOSEN, blind=blind_map)
            need = list(truth.values())
            need += [d.smiles.values[0] for d in res.values() if len(d)]
            need += [d.smiles.values[0] for d in resb.values() if len(d)]
            mkeys = _metric_keys(need)
            top_score, margin, correct = _top_stats(res, truth, mkeys)
            blind_top = np.array([float(d.score.values[0]) for d in resb.values() if len(d)])

            print('\n===== %s : P(rank-1 candidate correct | score bucket), class-1 queries =====' % tag)
            print(_bucket_table(top_score, correct, np.arange(0, 1.0001, 0.1), 'top score'))
            print('\n----- P(rank-1 correct | rank1-minus-rank2 margin) -----')
            print(_bucket_table(margin, correct, np.array([0, .02, .05, .1, .2, .3, .5, 1.0001]), 'margin'))
            print('\n----- class-2/3 simulation (every spectrum of the true ik14 removed) -----')
            print('top score  mean %.3f | p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f p95 %.3f p99 %.3f max %.3f'
                  % (blind_top.mean(), *np.percentile(blind_top, [10, 25, 50, 75, 90, 95, 99, 100])))
            print('class-1 top score   | p01 %.3f p05 %.3f p10 %.3f p25 %.3f p50 %.3f'
                  % tuple(np.percentile(top_score, [1, 5, 10, 25, 50])))
            print('\n  thr   class-1 kept   class-2/3 false alarm   precision @ 50/50 prior')
            for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                keep = float((top_score >= th).mean())
                fa = float((blind_top >= th).mean())
                prec = keep / (keep + fa) if (keep + fa) > 0 else float('nan')
                print('  %.2f      %.3f             %.3f                  %.3f' % (th, keep, fa, prec))
            out[tag] = dict(top_score=top_score.tolist(), margin=margin.tolist(),
                            correct=correct.tolist(), blind_top=blind_top.tolist())
        del lib
    with open(os.path.join(W, 'calibration.json'), 'w') as fh:
        json.dump(out, fh)
    return out


def stage_valsplit():
    """Split the shared validation set into genuine class 1 (the true structure keeps library
    spectra after its own library is excluded) and de-facto class 2/3 (it does not)."""
    S = pd.read_parquet(STRUCTS, columns=['ik14', 'smiles', 'mass', 'n_spectra', 'mkey'])
    D = pd.concat([pd.read_parquet(os.path.join(ROOT, 'work', '%s.parquet' % n))
                   for n in ('val_np', 'val_rand')], ignore_index=True)
    T = pd.concat([pd.read_csv(os.path.join(ROOT, 'work', '%s_truth.csv' % n))
                   for n in ('val_np', 'val_rand')], ignore_index=True)
    q = _to_queries(D, key='molecule_id')
    truth = dict(zip(T.molecule_id, T.smiles))
    xlib = {m: (l,) for m, l in zip(T.molecule_id, T.query_lib)}
    true_ik = dict(zip(T.molecule_id, T.ik14))
    masses = {k: estimate_molecule_mass([s['precursor_mz'] for s in v], [s['adduct'] for s in v],
                                        [s['ionization_mode'] for s in v]) for k, v in q.items()}
    lib = SpectralLibrary.load(os.path.join(W, 'lib_val.npz'))

    # does the true structure still have library spectra once its own library is removed?
    mk_of = dict(zip(S.ik14, S.mkey))
    same_mkey = S.groupby('mkey').ik14.apply(list).to_dict()
    row_of = {k: i for i, k in enumerate(lib.ik14)}
    has_ev = {}
    for mid in q:
        ik = true_ik[mid]
        sibs = same_mkey.get(mk_of.get(ik), [ik])
        n = 0
        for s_ik in sibs:
            r = row_of.get(s_ik)
            if r is None:
                continue
            ids = lib._spectra_of(np.array([r]))
            n += int((~np.isin(lib.spec_lib[ids], xlib[mid])).sum())
        has_ev[mid] = n > 0
    print('molecules whose true metric key keeps library spectra after excluding their own '
          'library: %d / %d' % (sum(has_ev.values()), len(has_ev)))

    res, dt = run_search(lib, q, masses, CHOSEN, xlib=xlib)
    s_all = score(res, truth)
    for label, sel in (('ALL              ', lambda m: True),
                       ('class 1 (in lib) ', lambda m: has_ev[m]),
                       ('class 2/3 (absent)', lambda m: not has_ev[m])):
        ids = [m for m in q if sel(m)]
        r = np.array([s_all['ranks'][m] for m in ids])
        rr = np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)
        print('  %s n=%4d  MRR %.4f  top1 %.3f  top5 %.3f  top25 %.3f'
              % (label, len(r), rr.mean(), (r == 1).mean(),
                 ((r > 0) & (r <= 5)).mean(), (r > 0).mean()))
    need = list(truth.values()) + [d.smiles.values[0] for d in res.values() if len(d)]
    mkeys = _metric_keys(need)
    c1 = [m for m in q if has_ev[m]]
    c23 = [m for m in q if not has_ev[m]]
    ts1 = np.array([float(res[m].score.values[0]) for m in c1 if len(res[m])])
    co1 = np.array([mkeys[res[m].smiles.values[0]] == mkeys[truth[m]] for m in c1 if len(res[m])])
    mg1 = np.array([float(res[m].score.values[0]) - (float(res[m].score.values[1]) if len(res[m]) > 1 else 0.0)
                    for m in c1 if len(res[m])])
    ts23 = np.array([float(res[m].score.values[0]) for m in c23 if len(res[m])])
    print('\n===== class-1 subset: P(rank-1 correct | score bucket), n=%d =====' % len(ts1))
    print(_bucket_table(ts1, co1, np.arange(0, 1.0001, 0.1), 'top score'))
    print('\n----- class-1 subset: P(rank-1 correct | margin) -----')
    print(_bucket_table(mg1, co1, np.array([0, .02, .05, .1, .2, .3, .5, 1.0001]), 'margin'))
    print('\n----- real class-2/3 subset (true structure genuinely absent), n=%d -----' % len(ts23))
    print('top score  mean %.3f | p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f p95 %.3f p99 %.3f max %.3f'
          % (ts23.mean(), *np.percentile(ts23, [10, 25, 50, 75, 90, 95, 99, 100])))
    print('class-1 top score | p01 %.3f p05 %.3f p10 %.3f p25 %.3f p50 %.3f'
          % tuple(np.percentile(ts1, [1, 5, 10, 25, 50])))
    print('\n  thr   class-1 kept   class-2/3 false alarm   precision @ observed prior (%.2f class 1)'
          % (len(ts1) / (len(ts1) + len(ts23))))
    pri = len(ts1) / (len(ts1) + len(ts23))
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        keep = float((ts1 >= th).mean())
        fa = float((ts23 >= th).mean())
        p = keep * pri / (keep * pri + fa * (1 - pri)) if (keep * pri + fa * (1 - pri)) else float('nan')
        print('  %.2f      %.3f             %.3f                  %.3f' % (th, keep, fa, p))
    json.dump(dict(class1_top=ts1.tolist(), class1_correct=co1.tolist(),
                   class1_margin=mg1.tolist(), class23_top=ts23.tolist()),
              open(os.path.join(W, 'calibration_split.json'), 'w'))

    # difficulty control: the hidden test molecules sit in the most crowded mass region
    # (median 80 candidates per +/-10 ppm window vs 9-13 here), so report MRR by window size.
    print('\n----- class-1 subset: MRR by number of candidates in the +/-10 ppm window -----')
    nc = {m: len(res[m]) for m in c1}
    print('%-14s %6s %8s %8s' % ('candidates', 'n', 'MRR', 'top1'))
    for a, b in ((1, 5), (5, 15), (15, 40), (40, 100), (100, 10 ** 9)):
        ids = [m for m in c1 if a <= nc[m] < b]
        if not ids:
            continue
        r = np.array([s_all['ranks'][m] for m in ids])
        rr = np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)
        print('%-14s %6d %8.4f %8.3f' % ('[%d,%s)' % (a, b if b < 10 ** 9 else 'inf'),
                                         len(r), rr.mean(), (r == 1).mean()))


def stage_val():
    """The shared validation sets (work/val_np*, work/val_rand*), using per-query exclude_libs.

    One library is built over ALL ingest_libs; each query excludes the library its own
    spectra came from.  This is the coordinator's class-1 protocol and covers 1079 molecules.
    """
    S = pd.read_parquet(STRUCTS, columns=['ik14', 'smiles', 'mass', 'n_spectra', 'mkey'])
    cache = os.path.join(W, 'lib_val.npz')
    specs, truths = [], []
    for name in ('val_np', 'val_rand'):
        d = pd.read_parquet(os.path.join(ROOT, 'work', '%s.parquet' % name))
        t = pd.read_csv(os.path.join(ROOT, 'work', '%s_truth.csv' % name))
        specs.append(d)
        truths.append(t)
    D = pd.concat(specs, ignore_index=True)
    T = pd.concat(truths, ignore_index=True)
    q = _to_queries(D, key='molecule_id')
    truth = dict(zip(T.molecule_id, T.smiles))
    xlib = {m: (l,) for m, l in zip(T.molecule_id, T.query_lib)}
    true_ik = dict(zip(T.molecule_id, T.ik14))
    masses = {k: estimate_molecule_mass([s['precursor_mz'] for s in v], [s['adduct'] for s in v],
                                        [s['ionization_mode'] for s in v]) for k, v in q.items()}
    if os.path.exists(cache):
        lib = SpectralLibrary.load(cache)
    else:
        wins = [mass_window(m, 15.0) for m in masses.values() if np.isfinite(m)]
        lib = SpectralLibrary.from_train(TRAIN, S, wins, verbose=False)
        lib.save(cache)
    print('library %d spectra / %d structures (all ingest_libs)' % (len(lib.spec_prec), len(lib.ik14)))

    # mass-window recall on this set
    sm = np.sort(S.mass.values)
    tm = dict(zip(S.ik14, S.mass))
    est = np.array([masses[m] for m in T.molecule_id])
    tru = np.array([tm[k] for k in T.ik14])
    for ppm in (5, 10, 15):
        lo, hi = est * (1 - ppm * 1e-6), est * (1 + ppm * 1e-6)
        nc = np.searchsorted(sm, hi, 'right') - np.searchsorted(sm, lo, 'left')
        print('  +/-%2d ppm: true-structure-in-window %.4f | candidates median %.0f p90 %.0f max %d'
              % (ppm, ((tru >= lo) & (tru <= hi)).mean(), np.median(nc), np.percentile(nc, 90), nc.max()))

    rows = _sweep(lib, q, truth, masses, COMBOS + FINE[:5], 'D(val_np+val_rand)', se=True, xlib=xlib)
    res, _ = run_search(lib, q, masses, CHOSEN, xlib=xlib)
    for o in ('random', 'n_spectra'):
        s = score(res, truth, order=o)
        s['se'] = bootstrap_se(s['ranks'])
        print(fmt('D(val_np+val_rand) | BASELINE %s' % o, s))
        rows.append(dict(tag='D', config='BASELINE ' + o, seconds=0.0,
                         **{k: v for k, v in s.items() if k != 'ranks'}))
    # per query_lib breakdown for the chosen config
    print('\nper query_lib (chosen config):')
    s_all = score(res, truth)
    for lb in sorted(set(T.query_lib)):
        ids = [m for m, l in zip(T.molecule_id, T.query_lib) if l == lb]
        r = np.array([s_all['ranks'][i] for i in ids])
        rr = np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)
        print('  %-20s n=%4d  MRR %.4f  top1 %.3f' % (lb, len(r), rr.mean(), (r == 1).mean()))

    # calibration on this larger set
    by_mkey = S.groupby('mkey').ik14.apply(list).to_dict()
    mk_of = dict(zip(S.ik14, S.mkey))
    blind_map = {m: by_mkey.get(mk_of.get(true_ik[m]), [true_ik[m]]) for m in q}
    resb, _ = run_search(lib, q, masses, CHOSEN, blind=blind_map, xlib=xlib)
    need = list(truth.values()) + [d.smiles.values[0] for d in res.values() if len(d)]
    mkeys = _metric_keys(need)
    top_score, margin, correct = _top_stats(res, truth, mkeys)
    blind_top = np.array([float(d.score.values[0]) for d in resb.values() if len(d)])
    print('\n===== D : P(rank-1 candidate correct | score bucket), n=%d =====' % len(top_score))
    print(_bucket_table(top_score, correct, np.arange(0, 1.0001, 0.1), 'top score'))
    print('\n----- P(rank-1 correct | rank1-minus-rank2 margin) -----')
    print(_bucket_table(margin, correct, np.array([0, .02, .05, .1, .2, .3, .5, 1.0001]), 'margin'))
    print('\n----- class-2/3 simulation (all structures sharing the truth metric key hidden) -----')
    print('top score  mean %.3f | p10 %.3f p25 %.3f p50 %.3f p75 %.3f p90 %.3f p95 %.3f p99 %.3f max %.3f'
          % (blind_top.mean(), *np.percentile(blind_top, [10, 25, 50, 75, 90, 95, 99, 100])))
    print('class-1 top score   | p01 %.3f p05 %.3f p10 %.3f p25 %.3f p50 %.3f'
          % tuple(np.percentile(top_score, [1, 5, 10, 25, 50])))
    print('\n  thr   class-1 kept   class-2/3 false alarm   precision @ 50/50 prior')
    for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        keep = float((top_score >= th).mean())
        fa = float((blind_top >= th).mean())
        print('  %.2f      %.3f             %.3f                  %.3f'
              % (th, keep, fa, keep / (keep + fa) if keep + fa else float('nan')))
    with open(os.path.join(W, 'calibration_val.json'), 'w') as fh:
        json.dump(dict(top_score=top_score.tolist(), margin=margin.tolist(),
                       correct=correct.tolist(), blind_top=blind_top.tolist()), fh)
    pd.DataFrame(rows).to_csv(os.path.join(W, 'grid_val.csv'), index=False)
    return rows


def stage_blend():
    """Does folding the library-popularity prior into the score help, or should the caller do it?"""
    for setname in ('np', 'gnps'):
        q, truth, masses, lib, S = load_set(setname)
        res, _ = run_search(lib, q, masses, CHOSEN)
        for alpha in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
            g = {}
            for mid, df in res.items():
                if len(df) == 0:
                    g[mid] = []
                    continue
                v = df.score.values + alpha * np.log1p(df.n_lib_spectra.values) / np.log(1000.0)
                g[mid] = df.smiles.values[np.argsort(-v, kind='stable')][:25].tolist()
            mrr, ranks = mrr_at_25(g, truth)
            r = np.array(list(ranks.values()))
            print('%-6s alpha=%.2f  MRR %.4f  top1 %.3f' % (setname, alpha, mrr, (r == 1).mean()))
        del lib


def _metric_keys(smiles_iter):
    from casmi.metric import cached_metric_key
    return {s: cached_metric_key(s) for s in set(smiles_iter)}


# ─────────────────────────────────────────────────────────────────────
# Stage: robustness
# ─────────────────────────────────────────────────────────────────────

def stage_robust():
    q, truth, masses, lib, S = load_set('gnps')
    print('gnps query set: %d molecules, %d spectra; library %d spectra / %d structures'
          % (len(q), sum(len(v) for v in q.values()), len(lib.spec_prec), len(lib.ik14)))
    rows = []
    for name, cfg in (('default', SearchConfig()), ('chosen', CHOSEN),
                      ('chosen+agg_max', CHOSEN.copy(agg='max')),
                      ('chosen+tol.02', CHOSEN.copy(tol_da=0.02)),
                      ('chosen+modified', CHOSEN.copy(modified=True))):
        res, dt = run_search(lib, q, masses, cfg)
        s = score(res, truth)
        print(fmt('C(gnps) | %s' % name, s, '(%.1fs)' % dt))
        rows.append(dict(tag='C(gnps)', config=name, seconds=dt,
                         **{k: v for k, v in s.items() if k != 'ranks'}))
    res, _ = run_search(lib, q, masses, CHOSEN)
    for o in ('random', 'n_spectra'):
        s = score(res, truth, order=o)
        print(fmt('C(gnps) | BASELINE %s' % o, s))
        rows.append(dict(tag='C(gnps)', config='BASELINE ' + o, seconds=0.0,
                         **{k: v for k, v in s.items() if k != 'ranks'}))
    pd.DataFrame(rows).to_csv(os.path.join(W, 'grid_gnps.csv'), index=False)
    return rows


# ─────────────────────────────────────────────────────────────────────
# Stage: timing
# ─────────────────────────────────────────────────────────────────────

def stage_timing():
    S = pd.read_parquet(STRUCTS, columns=['ik14', 'smiles', 'mass', 'n_spectra'])
    test = pd.read_parquet(os.path.join(ROOT, 'test.parquet'))
    q = _to_queries(test, key='molecule_id')
    masses = {k: estimate_molecule_mass([s['precursor_mz'] for s in v],
                                        [s['adduct'] for s in v],
                                        [s['ionization_mode'] for s in v]) for k, v in q.items()}
    wins = [mass_window(m, DEFAULT_MASS_PPM) for m in masses.values() if np.isfinite(m)]
    t0 = time.time()
    lib = SpectralLibrary.from_train(TRAIN, S, wins, verbose=False)
    t_build = time.time() - t0
    t0 = time.time()
    lib.set_config(CHOSEN)
    t_cfg = time.time() - t0
    for np_ in (1, 4):
        res, dt = run_search(lib, q, masses, CHOSEN, n_proc=np_)
        nc = np.array([len(d) for d in res.values()])
        print('real test set: %d molecules, search %.1f s with %d process(es); candidates '
              'med %d p90 %d max %d' % (len(q), dt, np_, np.median(nc), np.percentile(nc, 90), nc.max()))
    print('library build %.1f s (%d spectra / %d structures / %d peaks), set_config %.2f s'
          % (t_build, len(lib.spec_prec), len(lib.ik14), len(lib.peak_mz), t_cfg))
    return dict(build=t_build, config=t_cfg)


if __name__ == '__main__':
    stages = sys.argv[1:] or ['recall', 'grid', 'grid2', 'calib', 'robust', 'timing']
    for st in stages:
        print('\n' + '=' * 78 + '\n== %s\n' % st + '=' * 78)
        {'recall': stage_recall, 'grid': stage_grid, 'grid2': stage_grid2, 'fine': stage_fine,
         'calib': stage_calib, 'robust': stage_robust, 'timing': stage_timing, 'blend': stage_blend, 'val': stage_val, 'valsplit': stage_valsplit}[st]()
