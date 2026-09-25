"""
Produce submission.csv for a test.parquet. Same entry point locally and inside the Kaggle notebook.

    python -m casmi.run_test                                   # local: ./test.parquet -> output/submission.csv
"""
import os
import argparse
import pandas as pd

from casmi.common import molecule_neutral_mass, ppm_window
from casmi.pipeline import Pipeline, group_spectra, write_submission, load_ranker, log
from casmi.infer import ModelScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(test_path, sample_submission_path, train_path, assets_dir, out_path, n_proc=4, denovo_samples=128, use_pubchem=True):
    from casmi.libsearch import SpectralLibrary
    from casmi.candidates import CandidateDB
    test = pd.read_parquet(test_path)
    groups = group_spectra(test)
    log(f'test: {len(test)} spectra, {len(groups)} molecules')
    structures = pd.read_parquet(os.path.join(assets_dir, 'structures.parquet'))
    structures = structures[structures.mkey.notna()]
    scorer = ModelScorer(os.path.join(assets_dir, 'model.pt'), os.path.join(assets_dir, 'vocab.json'))
    log(f'model on {scorer.device} (bf16={scorer.amp})')
    windows = [ppm_window(molecule_neutral_mass([s['precursor_mz'] for s in g], [s['adduct'] for s in g]), 10.0) for g in groups.values()]
    library = SpectralLibrary.from_train(train_path, structures, windows)
    coconut = CandidateDB(os.path.join(assets_dir, 'coconut.parquet'), 'coconut')
    pub_path = os.path.join(assets_dir, 'pubchem.parquet')
    pubchem = CandidateDB(pub_path, 'pubchem') if use_pubchem and os.path.exists(pub_path) else None
    ranker = load_ranker(os.path.join(assets_dir, 'ranker.json'))
    pipe = Pipeline(scorer, structures, library, coconut, pubchem, ranker=ranker, cfg=dict(n_proc=n_proc, denovo_samples=denovo_samples))
    results = pipe.run(groups)
    sub = write_submission(results, sample_submission_path, out_path)
    n = sub.smiles.str.count(';') + 1
    log(f'wrote {len(sub)} rows to {out_path} | guesses per molecule: median {int(n.median())}, min {n.min()}, max {n.max()} '
        f'| fallback rows {(sub.smiles == "CCO").sum()}')
    return sub


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default=os.path.join(ROOT, 'test.parquet'))
    ap.add_argument('--sample', default=os.path.join(ROOT, 'sample_submission.csv'))
    ap.add_argument('--train', default=os.path.join(ROOT, 'train.parquet'))
    ap.add_argument('--assets', default=os.path.join(ROOT, 'kaggle_assets'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'output', 'submission.csv'))
    ap.add_argument('--n-proc', type=int, default=8)
    a = ap.parse_args()
    run(a.test, a.sample, a.train, a.assets, a.out, n_proc=a.n_proc)
