"""
End-to-end inference: test spectra -> up to 25 ranked SMILES per molecule.

Candidate sources (merged per molecule on the InChIKey first block, all restricted to the adduct-aware mass window):
  1. training structures whose library spectra resemble the query    (class 1: public spectra exist)
  2. COCONUT and PubChem structures in the mass window                (class 2: known structure, no spectra)
  3. de novo samples from the decoder whose mass matches              (class 3: novel structure)

Every candidate gets the same features (library similarity, predicted-fingerprint cosine, decoder
log-likelihood, source flags) and one linear ranker orders them. The same code path is used for local
validation and for the Kaggle run; validation only adds per-molecule exclusions to simulate the classes.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

from casmi.common import (
    molecule_neutral_mass, ppm_window, mol_from_smiles, morgan_bits, exact_mass, flat_canonical_smiles,
    plain_key, metric_key, _tautomer_enumerator, FP_BITS,
)
from rdkit import Chem

DEFAULT_CFG = dict(
    ppm_tol=10.0,            # candidate retrieval window around the calibrated neutral mass
    denovo_ppm=12.0,         # a generated structure must match the precursor mass to be kept
    denovo_samples=128,      # samples per spectrum (V3: doubled from 64)
    denovo_temperature=1.0,
    max_lib=200, max_coconut=150, max_pubchem=350, max_denovo=100,   # shortlist sizes sent to the decoder (V3: pubchem 150→350)
    n_proc=4,
    n_out=25,
)

FEATURES = ['fp_cos', 'fp_rank', 'll_rel', 'll_sqrt', 'll_tok', 'll_rank', 'lib_score', 'lib_hi', 'in_lib', 'in_coconut',
            'in_pubchem', 'db_only_pubchem', 'denovo_frac', 'denovo_any']


def add_features(t, n_samples):
    """Ranking features for one molecule's candidate table (needs fp_cos, ll, n_tok, lib_score, source flags, denovo_cnt)."""
    t['ll_rel'] = (t.ll - t.ll.max()).clip(lower=-40.0)
    t['ll_sqrt'] = -np.sqrt(-t.ll_rel)
    t['ll_tok'] = (t.ll / np.maximum(t.n_tok, 1)).clip(lower=-4.0)
    t['ll_rank'] = -np.log1p(t.ll.rank(ascending=False, method='min').values - 1)
    t['fp_rank'] = -np.log1p(t.fp_cos.rank(ascending=False, method='min').values - 1)
    t['lib_hi'] = (t.lib_score - 0.4).clip(lower=0.0)
    t['db_only_pubchem'] = ((t.in_pubchem == 1) & (t.in_coconut == 0) & (t.in_lib == 0)).astype(int)
    t['denovo_frac'] = t.denovo_cnt / n_samples
    t['denovo_any'] = (t.denovo_cnt > 0).astype(int)
    return add_group_features(t)


RANKER_FEATURES = FEATURES + ['lib_top', 'lib_margin_top', 'lib_hi7', 'lib_hi9', 'lib_conf', 'analog_sim', 'analog_tani', 'log_nspec']


def add_group_features(t):
    """Library-confidence features: a high similarity that also beats the runner-up by a clear margin is near-certain."""
    lib = t.lib_score.values.astype(np.float64)
    order = np.sort(lib)[::-1]
    best = order[0] if len(order) else 0.0
    second = order[1] if len(order) > 1 else 0.0
    top = (lib >= best) & (lib > 0)
    margin = np.where(top, best - second, 0.0)
    t = t.copy()
    t['lib_top'] = top.astype(int)
    t['lib_margin_top'] = margin
    t['lib_hi7'] = np.clip(lib - 0.7, 0, None)
    t['lib_hi9'] = np.clip(lib - 0.85, 0, None)
    t['lib_conf'] = top * lib * np.minimum(margin / 0.3, 1.0)
    # Analog evidence: if a library compound matches the spectra well but is not the answer, the answer is usually a
    # close structural relative of it (isomers give near-identical spectra). Tanimoto to the best library hits.
    analog_sim, analog_tani = np.zeros(len(t)), np.zeros(len(t))
    if 'fp_bits' in t.columns and len(t):
        bits = [set(b.tolist()) if b is not None else set() for b in t.fp_bits]
        hits = [j for j in np.argsort(-lib)[:3] if lib[j] > 0.3]
        for i, a in enumerate(bits):
            for rank, j in enumerate(hits):
                if j == i or not a or not bits[j]:
                    continue
                tani = len(a & bits[j]) / len(a | bits[j])
                analog_sim[i] = max(analog_sim[i], lib[j] * tani)
                if analog_tani[i] == 0.0:
                    analog_tani[i] = tani                      # similarity to the best hit that is not the candidate itself
    t['analog_sim'], t['analog_tani'] = analog_sim, analog_tani
    t['log_nspec'] = np.log1p(t.n_spectra.values.astype(np.float64)) if 'n_spectra' in t.columns else 0.0
    return t


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def group_spectra(df):
    """test.parquet-style frame -> {molecule_id: [spectrum dict, ...]} preserving first-appearance order."""
    groups = {}
    for row in df.itertuples(index=False):
        groups.setdefault(row.molecule_id, []).append(dict(
            mzs=row.ms2_mzs, intensities=row.ms2_normalized_intensities, precursor_mz=float(row.precursor_mz),
            adduct=row.adduct, ionization_mode=row.ionization_mode,
            collision_energy_ev=getattr(row, 'collision_energy_ev', None)))
    return groups


# ── worker functions (module level so they pickle) ──────────────────

def _fp_worker(smiles):
    return morgan_bits(smiles)


def _canon_worker(smiles):
    """-> (tautomer-canonical flat SMILES, metric key); (None, None) if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return (None, None)
    try:
        if mol.GetNumHeavyAtoms() <= 120:
            mol = _tautomer_enumerator().Canonicalize(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        return (smi, plain_key(mol))
    except Exception:
        return (flat_canonical_smiles(smiles), plain_key(smiles))


def _denovo_worker(args):
    smiles, mass, ppm = args
    mol = mol_from_smiles(smiles)
    if mol is None or '.' in smiles:
        return None
    m = exact_mass(mol)
    if not np.isfinite(m) or abs(m - mass) / mass * 1e6 > ppm:
        return None
    return (Chem.MolToSmiles(mol, isomericSmiles=False), plain_key(mol))


def _pool_map(fn, items, n_proc, chunksize=256):
    if len(items) == 0:
        return []
    if n_proc <= 1 or len(items) < 2000:
        return [fn(x) for x in items]
    with Pool(n_proc) as pool:
        return pool.map(fn, items, chunksize=chunksize)


def fp_cosine(pred, bit_lists):
    """Cosine between a predicted probability vector [4096] and binary fingerprints given as on-bit index arrays."""
    norm = float(np.linalg.norm(pred)) + 1e-9
    out = np.zeros(len(bit_lists), np.float32)
    for i, bits in enumerate(bit_lists):
        if bits is not None and len(bits):
            out[i] = pred[bits].sum() / (norm * np.sqrt(len(bits)))
    return out


class Pipeline:
    def __init__(self, scorer, structures, library=None, coconut=None, pubchem=None, ranker=None, cfg=None):
        self.scorer, self.library, self.coconut, self.pubchem = scorer, library, coconut, pubchem
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.ranker = ranker                      # dict(features=[...], coef=[...], intercept=float) or None
        st = structures.sort_values('mass').reset_index(drop=True)
        self.st, self.st_mass = st, st.mass.values
        self._model_cache, self._fp_cache, self._canon_cache = {}, {}, {}   # reused across runs (validation scenarios)

    # ── stage A: model passes that need only the spectra ──
    def _model_stage(self, groups, masses):
        cfg, out = self.cfg, {}
        jobs = []
        for mid, spectra in groups.items():
            states, mask = self.scorer.encode(spectra)
            fp = self.scorer.fingerprint(states, mask)
            samples = self.scorer.sample(states, mask, n_samples=cfg['denovo_samples'], temperature=cfg['denovo_temperature']) \
                if cfg['denovo_samples'] > 0 else []
            out[mid] = dict(fp=fp, n_samples=max(1, len(spectra) * cfg['denovo_samples']))
            best = {}
            for smi, lp in samples:
                cnt, mx = best.get(smi, (0, -np.inf))
                best[smi] = (cnt + 1, max(mx, lp))
            out[mid]['raw'] = best
            jobs += [(smi, masses[mid], cfg['denovo_ppm']) for smi in best]
        checked = _pool_map(_denovo_worker, jobs, cfg['n_proc'])
        pos = 0
        for mid in groups:
            agg = {}
            for smi, (cnt, lp) in out[mid].pop('raw').items():
                res = checked[pos]; pos += 1
                if res is None or res[1] is None:
                    continue
                canon, key = res
                c0, l0, _ = agg.get(key, (0, -np.inf, canon))
                agg[key] = (c0 + cnt, max(l0, lp), canon)
            out[mid]['denovo'] = agg
        return out

    # ── stage B: retrieval ──
    def _library_candidates(self, mid, spectra, lo, hi, exclude):
        i0, i1 = np.searchsorted(self.st_mass, lo), np.searchsorted(self.st_mass, hi, side='right')
        cand = self.st.iloc[i0:i1][['ik14', 'smiles', 'fp_bits', 'mkey', 'n_spectra']].copy()
        if exclude.get('ik14_lib'):
            cand = cand[~cand.ik14.isin(exclude['ik14_lib'])]
        cand['lib_score'] = 0.0
        if self.library is not None and len(cand):
            res = self.library.search(spectra, lo, hi, exclude_libs=tuple(exclude.get('libs', ())),
                                      exclude_ik14=tuple(exclude.get('ik14_lib', ())))
            if len(res):
                cand['lib_score'] = cand.ik14.map(dict(zip(res.ik14, res.score))).fillna(0.0).values
        return cand

    def run(self, groups, exclusions=None, return_features=False):
        cfg, exclusions = self.cfg, exclusions or {}
        mids = list(groups)
        masses = {m: molecule_neutral_mass([s['precursor_mz'] for s in groups[m]], [s['adduct'] for s in groups[m]]) for m in mids}
        windows = {m: ppm_window(masses[m], cfg['ppm_tol']) for m in mids}
        log(f'{len(mids)} molecules | model stage (fingerprints + de novo sampling)...')
        cache_key = (tuple(mids), cfg['denovo_samples'], cfg['denovo_temperature'])
        if cache_key not in self._model_cache:
            self._model_cache = {cache_key: self._model_stage(groups, masses)}
        model_out = self._model_cache[cache_key]

        log('retrieval stage...')
        db_frames = {}
        for name, db in (('coconut', self.coconut), ('pubchem', self.pubchem)):
            if db is not None:
                frames = db.query_many([windows[m] for m in mids], columns=['smiles', 'ik14'])
                db_frames[name] = dict(zip(mids, frames))

        tables = {}
        for mid in mids:
            ex = exclusions.get(mid, {})
            lo, hi = windows[mid]
            lib = self._library_candidates(mid, groups[mid], lo, hi, ex)
            rows = {k: dict(ik14=k, smiles=s, fp_bits=b, lib_score=l, in_lib=1, in_coconut=0, in_pubchem=0, denovo_cnt=0,
                            needs_canon=False, mkey=mk, n_spectra=ns)
                    for k, s, b, l, mk, ns in zip(lib.ik14, lib.smiles, lib.fp_bits, lib.lib_score, lib.mkey, lib.n_spectra)}
            for name in ('coconut', 'pubchem'):
                if name not in db_frames:
                    continue
                f = db_frames[name][mid]
                drop = ex.get('ik14_db') or ()
                for k, s in zip(f.ik14.values, f.smiles.values):
                    if k in drop:
                        continue
                    r = rows.get(k)
                    if r is None:
                        r = rows[k] = dict(ik14=k, smiles=s, fp_bits=None, lib_score=0.0, in_lib=0, in_coconut=0, in_pubchem=0,
                                           denovo_cnt=0, needs_canon=True, mkey=k, n_spectra=0)
                    r['in_' + name] = 1
            for k, (cnt, lp, canon) in model_out[mid]['denovo'].items():
                r = rows.get(k)
                if r is None:
                    r = rows[k] = dict(ik14=k, smiles=canon, fp_bits=None, lib_score=0.0, in_lib=0, in_coconut=0, in_pubchem=0,
                                       denovo_cnt=0, needs_canon=False, mkey=k, n_spectra=0)
                r['denovo_cnt'] = cnt
            tables[mid] = pd.DataFrame(list(rows.values())) if rows else pd.DataFrame(
                columns=['ik14', 'smiles', 'fp_bits', 'lib_score', 'in_lib', 'in_coconut', 'in_pubchem', 'denovo_cnt', 'needs_canon', 'mkey', 'n_spectra'])

        n_total = sum(len(t) for t in tables.values())
        log(f'fingerprint stage: {n_total:,} candidates...')
        todo = sorted({s for t in tables.values() for s, b in zip(t.smiles, t.fp_bits) if b is None} - self._fp_cache.keys())
        self._fp_cache.update(zip(todo, _pool_map(_fp_worker, todo, cfg['n_proc'], chunksize=2000)))
        bits = self._fp_cache
        for mid, t in tables.items():
            if len(t) == 0:
                continue
            t['fp_bits'] = [b if b is not None else bits.get(s) for s, b in zip(t.smiles, t.fp_bits)]
            t['fp_cos'] = fp_cosine(model_out[mid]['fp'], t.fp_bits.tolist())

        log('shortlist + canonicalisation stage...')
        for mid, t in tables.items():
            if len(t) == 0:
                continue
            t['log_ncand'] = np.log1p(len(t))
            keep = np.zeros(len(t), bool)
            pre = t.fp_cos.values + t.lib_score.values
            for flag, cap in (('in_lib', cfg['max_lib']), ('in_coconut', cfg['max_coconut']), ('in_pubchem', cfg['max_pubchem'])):
                idx = np.flatnonzero(t[flag].values == 1)
                keep[idx[np.argsort(-pre[idx])[:cap]]] = True
            idx = np.flatnonzero(t.denovo_cnt.values > 0)
            keep[idx[np.argsort(-t.denovo_cnt.values[idx])[:cfg['max_denovo']]]] = True
            tables[mid] = t[keep].reset_index(drop=True)
        todo = sorted({s for t in tables.values() if len(t) for s, n in zip(t.smiles, t.needs_canon) if n} - self._canon_cache.keys())
        self._canon_cache.update(zip(todo, _pool_map(_canon_worker, todo, cfg['n_proc'], chunksize=64)))
        canon = self._canon_cache
        for mid, t in tables.items():
            if len(t) == 0:
                continue
            t['score_smiles'] = [(canon[s][0] or s) if n else s for s, n in zip(t.smiles, t.needs_canon)]
            t['mkey'] = [(canon[s][1] or k) if n else k for s, n, k in zip(t.smiles, t.needs_canon, t.mkey)]

        log('decoder likelihood stage...')
        for mid, t in tables.items():
            if len(t) == 0:
                continue
            states, mask = self.scorer.encode(groups[mid])
            ll, n_tok = self.scorer.loglik(states, mask, t.score_smiles.tolist())
            finite = np.isfinite(ll)
            floor = (ll[finite].min() - 5.0) if finite.any() else -100.0
            t['ll'] = np.where(finite, ll, floor)
            t['n_tok'] = n_tok
            tables[mid] = t = add_features(t, model_out[mid]['n_samples'])

        log('ranking stage...')
        results = {}
        for mid, t in tables.items():
            if len(t) == 0:
                results[mid] = []
                continue
            t['final'] = self.rank_score(t)
            t.sort_values('final', ascending=False, inplace=True, kind='stable')
            t.reset_index(drop=True, inplace=True)
        # dedupe the head of each list on the real metric key (tautomer-canonical)
        head = {mid: t.score_smiles.head(3 * cfg['n_out']).tolist() for mid, t in tables.items() if len(t)}
        todo = sorted({s for v in head.values() for s in v} - self._canon_cache.keys())
        self._canon_cache.update(zip(todo, _pool_map(_canon_worker, todo, cfg['n_proc'], chunksize=64)))
        keyed = self._canon_cache
        for mid, smiles in head.items():
            seen, out = set(), []
            for s in smiles:
                canon_smi, key = keyed.get(s, (None, None))
                if canon_smi is None or key in seen:
                    continue
                seen.add(key); out.append(canon_smi)
                if len(out) == cfg['n_out']:
                    break
            results[mid] = out
        log('done')
        return (results, tables) if return_features else results

    def rank_score(self, t):
        if self.ranker is None:     # untuned fallback: library evidence first, then the two model scores
            return 2.0 * t.lib_score.values + t.fp_cos.values + 0.05 * t.ll_rel.values + 0.3 * t.in_lib.values \
                + 0.15 * t.in_coconut.values
        r = self.ranker
        x = t[r['features']].values.astype(np.float64)
        x = (x - np.asarray(r['mean'])) / np.asarray(r['scale'])
        if 'coef' in r:
            return x @ np.asarray(r['coef'])
        w = {k: np.asarray(v) for k, v in r['weights'].items()}
        if r.get('hidden'):
            h = np.tanh(x @ w['0.weight'].T + w['0.bias'])
            return (h @ w['2.weight'].T + w['2.bias'])[:, 0]
        return (x @ w['weight'].T)[:, 0]


def write_submission(results, sample_submission_path, out_path, fallback='CCO'):
    sub = pd.read_csv(sample_submission_path)
    sub['smiles'] = [';'.join(results.get(m, [])[:25]) or fallback for m in sub.molecule_id]
    sub.to_csv(out_path, index=False)
    return sub


def load_ranker(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None
