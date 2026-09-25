"""Run the pipeline on the validation sets under every simulated class and save the candidate feature tables
(work/feats_<set>.parquet) so ranker designs can be compared offline without touching the GPU again."""
import os
import argparse
import pandas as pd

from casmi.pipeline import group_spectra, RANKER_FEATURES, log
from casmi.validate import build_pipeline, exclusions_for, report, WORK

KEEP = RANKER_FEATURES + ['lib_score', 'fp_cos', 'll', 'n_tok', 'denovo_cnt', 'mkey', 'score_smiles']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='val_np,val_rand')
    ap.add_argument('--scenarios', default='c1,c2,c3')
    ap.add_argument('--tag', default='')
    ap.add_argument('--n-proc', type=int, default=8)
    ap.add_argument('--denovo-samples', type=int, default=64)
    ap.add_argument('--no-pubchem', action='store_true')
    ap.add_argument('--model', default=None)
    args = ap.parse_args()
    args.no_library = args.no_db = False
    args.ranker = None
    names = args.sets.split(',')
    frames = {n: pd.read_parquet(os.path.join(WORK, f'{n}.parquet')) for n in names}
    truths = {n: pd.read_csv(os.path.join(WORK, f'{n}_truth.csv')) for n in names}
    pipe = build_pipeline(args, list(frames.values()))
    mkey_of = dict(zip(pipe.st.ik14, pipe.st.mkey))
    for name in names:
        groups, truth, rows = group_spectra(frames[name]), truths[name], []
        for sc in args.scenarios.split(','):
            results, tables = pipe.run(groups, exclusions_for(truth, sc, pipe.st), return_features=True)
            report(f'[fallback ranker] {name} {sc}', results, truth)
            for r in truth.itertuples(index=False):
                t = tables[r.molecule_id]
                if len(t) == 0:
                    continue
                t = t[list(dict.fromkeys(KEEP))].copy()
                t['label'] = (t.mkey == mkey_of.get(r.ik14, r.ik14)).astype(int)
                t['molecule_id'], t['scenario'], t['set'] = r.molecule_id, sc, name
                rows.append(t)
        out = os.path.join(WORK, f'feats_{name}{args.tag}.parquet')
        pd.concat(rows, ignore_index=True).to_parquet(out, index=False)
        log('wrote', out)


if __name__ == '__main__':
    main()
