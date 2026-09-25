"""
V2 training data: randomized-SMILES variants for every training structure and a structure-only set from COCONUT.

    python -m casmi.prep_v2   ->  work/v2/{vocab.json, train_tokens.npy, coco_tokens.npy, coco_mass.npy, coco_fp.npy}

Why: the V1 run memorised training SMILES strings (validation loss on unseen structures bottomed at epoch 2).
Randomized SMILES make string memorisation useless, and mass-conditioned structure-only examples from COCONUT
teach the decoder natural-product chemistry it has no spectra for. Validation structures are excluded from COCONUT.
"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem

from casmi.common import SmilesTokenizer, morgan_bits, mol_from_smiles, split_smiles, PAD_ID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')
OUT = os.path.join(WORK, 'v2')
MAX_TOKENS, N_TRAIN_VAR, N_COCO_VAR, FP_PAD = 160, 4, 2, 128


def _variants(args):
    smiles, n, want_fp = args
    mol = mol_from_smiles(smiles)
    if mol is None:
        return [], None
    out = []
    for _ in range(n):
        try:
            out.append(Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=False))
        except Exception:
            pass
    return out, (morgan_bits(mol) if want_fp else None)


def _encode_matrix(tokenizer, canon, variants, n_var):
    mat = np.zeros((len(canon), n_var + 1, MAX_TOKENS), np.int16)
    ok = np.zeros(len(canon), bool)
    for i, (c, vs) in enumerate(zip(canon, variants)):
        ids = tokenizer.encode(c)
        if ids is None or len(ids) > MAX_TOKENS:
            continue
        ok[i] = True
        mat[i, :, :len(ids)] = ids                       # default every slot to the canonical string
        for j, v in enumerate(vs[:n_var]):
            vid = tokenizer.encode(v)
            if vid is not None and len(vid) <= MAX_TOKENS:
                mat[i, j + 1, :] = PAD_ID
                mat[i, j + 1, :len(vid)] = vid
    return mat, ok


def main():
    os.makedirs(OUT, exist_ok=True)
    structs = pd.read_parquet(os.path.join(WORK, 'structures.parquet'), columns=['ik14', 'smiles', 'mkey', 'split', 'trainable'])
    coco = pd.read_parquet(os.path.join(WORK, 'db', 'coconut.parquet'))
    val = structs[structs.split != 'train']
    banned = set(val.ik14) | set(val.mkey.dropna())
    coco = coco[~coco.ik14.isin(banned)].reset_index(drop=True)
    print(f'train structures {len(structs):,} | COCONUT after removing validation structures {len(coco):,}', flush=True)

    with Pool(10) as pool:
        tv = pool.map(_variants, [(s, N_TRAIN_VAR, False) for s in structs.smiles], chunksize=1000)
        cv = pool.map(_variants, [(s, N_COCO_VAR, True) for s in coco.smiles], chunksize=1000)
    train_var = [v for v, _ in tv]
    coco_var, coco_bits = [v for v, _ in cv], [b for _, b in cv]

    def corpus():
        yield from structs.smiles
        yield from coco.smiles
        for vs in train_var:
            yield from vs
        for vs in coco_var:
            yield from vs
    tokenizer = SmilesTokenizer.build(corpus(), min_count=50)
    tokenizer.save(os.path.join(OUT, 'vocab.json'))
    print('vocab', len(tokenizer), tokenizer.itos, flush=True)

    train_tok, train_ok = _encode_matrix(tokenizer, structs.smiles.tolist(), train_var, N_TRAIN_VAR)
    lost = int((structs.trainable.values & ~train_ok).sum())
    print(f'train structures encodable {train_ok.sum():,}; previously-trainable now lost: {lost}')
    np.save(os.path.join(OUT, 'train_tokens.npy'), train_tok)
    np.save(os.path.join(OUT, 'train_ok.npy'), train_ok)

    coco_tok, coco_ok = _encode_matrix(tokenizer, coco.smiles.tolist(), coco_var, N_COCO_VAR)
    coco_ok &= np.array([b is not None for b in coco_bits])
    fp = np.full((len(coco), FP_PAD), -1, np.int16)
    for i, b in enumerate(coco_bits):
        if b is not None:
            fp[i, :min(len(b), FP_PAD)] = b[:FP_PAD]
    np.save(os.path.join(OUT, 'coco_tokens.npy'), coco_tok[coco_ok])
    np.save(os.path.join(OUT, 'coco_mass.npy'), coco.mass.values[coco_ok])
    np.save(os.path.join(OUT, 'coco_fp.npy'), fp[coco_ok])
    print(f'COCONUT structure-only examples: {int(coco_ok.sum()):,}')
    frac_diff = np.mean([(train_tok[i, 1] != train_tok[i, 0]).any() for i in np.flatnonzero(train_ok)[:5000]])
    print(f'fraction of structures whose first random variant differs from canonical: {frac_diff:.3f}')


if __name__ == '__main__':
    main()
