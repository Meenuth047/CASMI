"""
Data preparation (CPU only).

  python -m casmi.prep structures   -> work/structures.parquet, work/vocab.json
  python -m casmi.prep spectra      -> work/spectra/shard_XX.npz  (compact training arrays)

structures.parquet has one row per unique training structure (inchikey14) with its exact mass,
formula, tautomer-canonical metric key, token ids, Morgan on-bits and the train/val split.
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from multiprocessing import Pool

from casmi.common import (
    TEST_ADDUCTS, ADDUCT_TO_ID, SmilesTokenizer, exact_mass, mol_formula, mol_from_smiles,
    metric_key, plain_key, morgan_bits, prep_peaks, first_collision_energy, MAX_PEAKS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, 'train.parquet')
WORK = os.path.join(ROOT, 'work')
STRUCT_PATH = os.path.join(WORK, 'structures.parquet')
VOCAB_PATH = os.path.join(WORK, 'vocab.json')
SPECTRA_DIR = os.path.join(WORK, 'spectra')

N_PROC = 10                 # leave headroom for the other training job on this machine
MAX_TOKENS = 160            # BOS + tokens + EOS; longer structures are not used for training
MAX_PPM_ERROR = 20.0        # label-quality filter on precursor_error_ppm
MAX_HEAVY_FOR_TAUTOMER = 120
N_VAL_RANDOM = 1000
LIBS = ['enveda-180', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse',
        'msdial', 'drug_plus', 'enveda-np-examples', 'masaryk']
LIB_TO_ID = {lib: i for i, lib in enumerate(LIBS)}


def _struct_worker(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return (float('nan'), None, None, None)
    mkey = metric_key(mol) if mol.GetNumHeavyAtoms() <= MAX_HEAVY_FOR_TAUTOMER else plain_key(mol)
    bits = morgan_bits(mol)
    return (exact_mass(mol), mol_formula(mol), mkey, bits)


def build_structures():
    os.makedirs(WORK, exist_ok=True)
    print('reading structure columns...', flush=True)
    df = pq.read_table(TRAIN_PATH, columns=['ingest_lib', 'normalized_smiles', 'inchikey14']).to_pandas()
    n_spectra = df.groupby('inchikey14').size()
    in_np = set(df.loc[df.ingest_lib == 'enveda-np-examples', 'inchikey14'])
    np_like = set(df.loc[df.ingest_lib.isin(['gnps', 'riken', 'mona', 'massbank', 'msdial', 'spectraverse']), 'inchikey14'])
    structs = df.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles']].reset_index(drop=True)
    structs.columns = ['ik14', 'smiles']
    del df
    print(f'{len(structs):,} unique structures; computing mass / metric key / fingerprints...', flush=True)

    with Pool(N_PROC) as pool:
        out = pool.map(_struct_worker, structs.smiles.tolist(), chunksize=500)
    structs['mass'] = [o[0] for o in out]
    structs['formula'] = [o[1] for o in out]
    structs['mkey'] = [o[2] for o in out]
    structs['fp_bits'] = [o[3] for o in out]
    structs['n_spectra'] = structs.ik14.map(n_spectra).astype(np.int32)

    tokenizer = SmilesTokenizer.build(structs.smiles[structs.mkey.notna()], min_count=5)
    tokenizer.save(VOCAB_PATH)
    structs['tokens'] = [tokenizer.encode(s) for s in structs.smiles]
    n_tok = structs.tokens.map(lambda t: len(t) if t is not None else 0)
    structs['trainable'] = structs.mkey.notna() & (n_tok > 0) & (n_tok <= MAX_TOKENS)
    print(f'vocab {len(tokenizer)} tokens; trainable structures {structs.trainable.sum():,} / {len(structs):,}')

    # Splits. val_np: half of the enveda-np-examples structures (the library closest to the hidden
    # test set) are held out of model training entirely. val_rand: random natural-product-library
    # structures, also held out. Everything else trains.
    rng = np.random.default_rng(0)
    split = np.array(['train'] * len(structs), dtype=object)
    np_idx = np.flatnonzero(structs.ik14.isin(in_np).values & structs.trainable.values)
    split[rng.choice(np_idx, size=len(np_idx) // 2, replace=False)] = 'val_np'
    rand_pool = np.flatnonzero(structs.ik14.isin(np_like).values & structs.trainable.values & (split == 'train')
                               & ~structs.ik14.isin(in_np).values)
    split[rng.choice(rand_pool, size=N_VAL_RANDOM, replace=False)] = 'val_rand'
    structs['split'] = split
    print(structs.split.value_counts().to_string())

    structs.to_parquet(STRUCT_PATH, index=False)
    print('wrote', STRUCT_PATH, flush=True)


def _spectra_worker(rg_idx):
    structs = pd.read_parquet(STRUCT_PATH, columns=['ik14', 'trainable'])
    struct_index = {k: i for i, k in enumerate(structs.ik14)}
    trainable = structs.trainable.values
    cols = ['ingest_lib', 'inchikey14', 'adduct', 'precursor_mz', 'precursor_error_ppm',
            'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']
    t = pq.ParquetFile(TRAIN_PATH).read_row_group(rg_idx, columns=cols).to_pandas()
    keep = t.adduct.isin(TEST_ADDUCTS) & (t.precursor_error_ppm.abs() <= MAX_PPM_ERROR)
    t = t[keep]
    n = len(t)
    mz = np.zeros((n, MAX_PEAKS), np.float32)
    inten = np.zeros((n, MAX_PEAKS), np.float16)
    n_peaks = np.zeros(n, np.int16)
    prec = np.zeros(n, np.float64)
    adduct = np.zeros(n, np.int8)
    ce = np.full(n, -1.0, np.float32)
    lib = np.zeros(n, np.int8)
    sidx = np.zeros(n, np.int32)
    j = 0
    for row in t.itertuples(index=False):
        si = struct_index.get(row.inchikey14)
        if si is None or not trainable[si]:
            continue
        m, it = prep_peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz)
        if len(m) == 0:
            continue
        mz[j, :len(m)], inten[j, :len(m)], n_peaks[j] = m, it, len(m)
        prec[j], adduct[j], lib[j], sidx[j] = row.precursor_mz, ADDUCT_TO_ID[row.adduct], LIB_TO_ID[row.ingest_lib], si
        c = first_collision_energy(row.collision_energy_ev)
        ce[j] = c if np.isfinite(c) else -1.0
        j += 1
    path = os.path.join(SPECTRA_DIR, f'shard_{rg_idx:02d}.npz')
    np.savez(path, mz=mz[:j], inten=inten[:j], n_peaks=n_peaks[:j], prec=prec[:j], adduct=adduct[:j],
             ce=ce[:j], lib=lib[:j], sidx=sidx[:j])
    return rg_idx, j


def build_spectra():
    os.makedirs(SPECTRA_DIR, exist_ok=True)
    n_rg = pq.ParquetFile(TRAIN_PATH).metadata.num_row_groups
    total = 0
    with Pool(6) as pool:
        for rg_idx, n in pool.imap_unordered(_spectra_worker, range(n_rg)):
            total += n
            print(f'row group {rg_idx:2d}: kept {n:,}', flush=True)
    print(f'total training-eligible spectra: {total:,}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('structures', 'all'):
        build_structures()
    if cmd in ('spectra', 'all'):
        build_spectra()
