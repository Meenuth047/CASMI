"""
Local replica of the competition metric: MRR@25 on tautomer-canonical InChIKey first blocks.

    from casmi.metric import score_submission, mrr_at_25
    result = score_submission(submission_df, truth)      # truth: {molecule_id: smiles}
    print(result['mrr'])

CLI:  python -m casmi.metric submission.csv truth.csv     (truth.csv: molecule_id,smiles)
"""

import sys
import numpy as np
import pandas as pd
from functools import lru_cache

from casmi.common import metric_key

MAX_GUESSES = 25


@lru_cache(maxsize=2_000_000)
def cached_metric_key(smiles):
    return metric_key(smiles)


def validate_submission(sub, expected_ids=None):
    """Raise ValueError for anything Kaggle would reject."""
    for col in ('molecule_id', 'smiles'):
        if col not in sub.columns:
            raise ValueError(f'missing column {col}')
    if len(sub) == 0:
        raise ValueError('empty submission')
    if sub.molecule_id.isna().any() or sub.smiles.isna().any():
        raise ValueError('nulls present')
    if sub.molecule_id.duplicated().any():
        raise ValueError('repeated molecule_id')
    too_many = sub.smiles.astype(str).str.count(';') + 1 > MAX_GUESSES
    if too_many.any():
        raise ValueError(f'{int(too_many.sum())} rows have more than {MAX_GUESSES} guesses')
    if expected_ids is not None:
        missing = set(expected_ids) - set(sub.molecule_id)
        extra = set(sub.molecule_id) - set(expected_ids)
        if missing or extra:
            raise ValueError(f'{len(missing)} molecule_ids missing, {len(extra)} unexpected')
    return True


def first_correct_rank(guesses, truth_key):
    """1-based rank of the first guess whose metric key equals truth_key; 0 if none in the top 25."""
    for rank, smi in enumerate(guesses[:MAX_GUESSES], start=1):
        if smi and cached_metric_key(smi) == truth_key:
            return rank
    return 0


def mrr_at_25(ranked_guesses, truth):
    """ranked_guesses: {molecule_id: [smiles, ...]}, truth: {molecule_id: smiles}. Returns (mrr, ranks)."""
    ranks = {}
    for mol_id, true_smiles in truth.items():
        truth_key = cached_metric_key(true_smiles)
        ranks[mol_id] = first_correct_rank(list(ranked_guesses.get(mol_id, [])), truth_key) if truth_key else 0
    rr = [1.0 / r if r > 0 else 0.0 for r in ranks.values()]
    return (float(np.mean(rr)) if rr else 0.0), ranks


def score_submission(sub, truth):
    validate_submission(sub)
    guesses = {m: str(s).split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
    mrr, ranks = mrr_at_25(guesses, truth)
    r = np.array(list(ranks.values()))
    return {
        'mrr': mrr,
        'n': len(r),
        'top1': float((r == 1).mean()) if len(r) else 0.0,
        'top5': float(((r > 0) & (r <= 5)).mean()) if len(r) else 0.0,
        'top25': float((r > 0).mean()) if len(r) else 0.0,
        'ranks': ranks,
    }


def summarize(result):
    return (f"MRR@25 {result['mrr']:.4f} | top1 {result['top1']:.3f} | top5 {result['top5']:.3f} "
            f"| top25 {result['top25']:.3f} | n={result['n']}")


if __name__ == '__main__':
    sub = pd.read_csv(sys.argv[1])
    truth_df = pd.read_csv(sys.argv[2])
    print(summarize(score_submission(sub, dict(zip(truth_df.molecule_id, truth_df.smiles)))))
