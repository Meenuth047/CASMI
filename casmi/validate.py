"""
Score the full pipeline on held-out molecules under the three novelty classes, and fit the ranker.

    python -m casmi.validate --set val_np                      # report MRR@25 per simulated class
    python -m casmi.validate --set val_rand --fit-ranker       # fit work/ranker.json on val_rand, then report on val_np

Class simulation (the model never trained on these molecules in any case):
    c1  library keeps the molecule's spectra from OTHER source libraries (the query's own library is excluded)
    c2  molecule removed from the spectral library; still retrievable from COCONUT / PubChem if it is there
    c3  molecule removed from the library AND the databases; only de novo generation can find it
"""
import os
import json
import argparse
import numpy as np
import pandas as pd

from casmi.pipeline import Pipeline, group_spectra, load_ranker, FEATURES, log
from casmi.metric import mrr_at_25
from casmi.infer import ModelScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')


def build_pipeline(args, query_frames):
    from casmi.common import molecule_neutral_mass, ppm_window
    if not args.no_library:
        from casmi.libsearch import SpectralLibrary
    if not args.no_db:
        from casmi.candidates import CandidateDB
    structures = pd.read_parquet(os.path.join(WORK, 'structures.parquet'))
    structures = structures[structures.mkey.notna()]
    scorer = ModelScorer(getattr(args, 'model', None) or os.path.join(WORK, 'ckpt', 'model.pt'), os.path.join(WORK, 'vocab.json'))
    windows = []
    for df in query_frames:
        for _, g in df.groupby('molecule_id'):
            windows.append(ppm_window(molecule_neutral_mass(g.precursor_mz.tolist(), g.adduct.tolist()), 10.0))
    library = SpectralLibrary.from_train(os.path.join(ROOT, 'train.parquet'), structures, windows) if not args.no_library else None
    coconut = CandidateDB(os.path.join(WORK, 'db', 'coconut.parquet'), 'coconut') if not args.no_db else None
    pub_path = os.path.join(WORK, 'db', 'pubchem.parquet')
    pubchem = CandidateDB(pub_path, 'pubchem') if (not args.no_db and not args.no_pubchem and os.path.exists(pub_path)) else None
    cfg = dict(n_proc=args.n_proc, denovo_samples=args.denovo_samples)
    return Pipeline(scorer, structures, library, coconut, pubchem, ranker=load_ranker(args.ranker), cfg=cfg)


def exclusions_for(truth, scenario, structures):
    # hide every training structure that shares the truth's metric key (tautomers collapse to one answer)
    mkey_of = dict(zip(structures.ik14, structures.mkey))
    by_mkey = structures.groupby('mkey').ik14.agg(set).to_dict()
    ex = {}
    for r in truth.itertuples(index=False):
        same = set(by_mkey.get(mkey_of.get(r.ik14), set())) | {r.ik14}
        if scenario == 'c1':
            ex[r.molecule_id] = {'libs': [r.query_lib]}
        elif scenario == 'c2':
            ex[r.molecule_id] = {'libs': [r.query_lib], 'ik14_lib': same}
        else:
            ex[r.molecule_id] = {'libs': [r.query_lib], 'ik14_lib': same, 'ik14_db': same}
    return ex


def fit_listwise(feats, l2=1e-3, steps=400, scenario_weights=None):
    """Linear listwise ranker: maximise the softmax probability of the correct candidate within each molecule's list."""
    import torch
    scenario_weights = scenario_weights or {}
    x = feats[FEATURES].values.astype(np.float64)
    mean, scale = x.mean(0), x.std(0) + 1e-9
    x = torch.tensor((x - mean) / scale, dtype=torch.float32)
    y = torch.tensor(feats.label.values, dtype=torch.float32)
    gid_raw = (feats.molecule_id + '|' + feats.scenario).values
    codes, uniq = pd.factorize(gid_raw)
    gid = torch.tensor(codes)
    n_groups = len(uniq)
    has_pos = torch.zeros(n_groups).index_add_(0, gid, y) > 0
    gw = torch.tensor([scenario_weights.get(u.split('|')[1], 1.0) for u in uniq], dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        s = x @ w
        m = torch.full((n_groups,), -1e9).scatter_reduce(0, gid, s, reduce='amax')       # stabiliser per group
        e = torch.exp(s - m[gid])
        denom = torch.zeros(n_groups).index_add_(0, gid, e)
        numer = torch.zeros(n_groups).index_add_(0, gid, e * y)
        nll = -(torch.log(numer[has_pos] + 1e-12) - torch.log(denom[has_pos]))
        loss = (nll * gw[has_pos]).sum() / gw[has_pos].sum() + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    log(f'listwise ranker fitted on {int(has_pos.sum())} lists (of {n_groups}); final loss {closure().item():.4f}')
    return dict(features=FEATURES, mean=mean.tolist(), scale=scale.tolist(), coef=w.detach().numpy().astype(float).tolist())


def report(name, results, truth):
    mrr, ranks = mrr_at_25(results, dict(zip(truth.molecule_id, truth.smiles)))
    r = np.array(list(ranks.values()))
    log(f'{name}: MRR@25 {mrr:.4f} | top1 {(r == 1).mean():.3f} | top5 {((r > 0) & (r <= 5)).mean():.3f} | top25 {(r > 0).mean():.3f} | n={len(r)}')
    return mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='val_np')
    ap.add_argument('--scenarios', default='c1,c2,c3')
    ap.add_argument('--fit-ranker', action='store_true')
    ap.add_argument('--ranker', default=os.path.join(WORK, 'ranker.json'))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--n-proc', type=int, default=8)
    ap.add_argument('--denovo-samples', type=int, default=64)
    ap.add_argument('--no-library', action='store_true')
    ap.add_argument('--no-db', action='store_true')
    ap.add_argument('--no-pubchem', action='store_true')
    args = ap.parse_args()

    sets = [args.set] + (['val_np'] if args.fit_ranker and args.set != 'val_np' else [])
    frames, truths = {}, {}
    for name in sets:
        df = pd.read_parquet(os.path.join(WORK, f'{name}.parquet'))
        truth = pd.read_csv(os.path.join(WORK, f'{name}_truth.csv'))
        if args.limit:
            truth = truth.head(args.limit)
            df = df[df.molecule_id.isin(truth.molecule_id)]
        frames[name], truths[name] = df, truth
    if args.fit_ranker:
        args.ranker = None if not os.path.exists(args.ranker) else args.ranker
    pipe = build_pipeline(args, list(frames.values()))
    scenarios = args.scenarios.split(',')

    if args.fit_ranker:
        pipe.ranker = None
        name, rows = args.set, []
        groups, truth = group_spectra(frames[name]), truths[name]
        mkey_truth = pd.read_parquet(os.path.join(WORK, 'structures.parquet'), columns=['ik14', 'mkey']).set_index('ik14').mkey
        for sc in scenarios:
            results, tables = pipe.run(groups, exclusions_for(truth, sc, pipe.st), return_features=True)
            report(f'[untuned] {name} {sc}', results, truth)
            for r in truth.itertuples(index=False):
                t = tables[r.molecule_id]
                if len(t) == 0:
                    continue
                t = t[FEATURES + ['mkey']].copy()
                t['label'] = (t.mkey == mkey_truth.get(r.ik14, r.ik14)).astype(int)
                t['molecule_id'], t['scenario'] = r.molecule_id, sc
                rows.append(t)
        feats = pd.concat(rows, ignore_index=True)
        feats.to_parquet(os.path.join(WORK, 'ranker_features.parquet'), index=False)
        ranker = fit_listwise(feats)
        with open(os.path.join(WORK, 'ranker.json'), 'w') as fh:
            json.dump(ranker, fh, indent=1)
        log('ranker coefficients: ' + ', '.join(f'{f}={c:+.2f}' for f, c in zip(FEATURES, ranker['coef'])))
        pipe.ranker = ranker
        eval_name = 'val_np'
    else:
        eval_name = args.set

    groups, truth = group_spectra(frames[eval_name]), truths[eval_name]
    summary = {}
    for sc in scenarios:
        results = pipe.run(groups, exclusions_for(truth, sc, pipe.st))
        summary[sc] = report(f'{eval_name} {sc}', results, truth)
    log('summary ' + json.dumps({k: round(v, 4) for k, v in summary.items()}))


if __name__ == '__main__':
    main()
