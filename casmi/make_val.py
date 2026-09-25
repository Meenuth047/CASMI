"""
Build held-out validation query sets in test.parquet format (+ truth):

  work/val_np.parquet    125 enveda-np-examples molecules (timsTOF, same pipeline as the hidden test) that the model never trains on
  work/val_rand.parquet  1000 random natural-product-library molecules, queries drawn from ONE library per molecule
  work/val_*_truth.csv   molecule_id, smiles, ik14, query_lib
"""
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.common import TEST_ADDUCTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')
TEST_COLS = ['ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct', 'ionization_mode',
             'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev', 'collision_energy_orig_units']


def main(max_spectra=6):
    structs = pd.read_parquet(os.path.join(WORK, 'structures.parquet'), columns=['ik14', 'smiles', 'split'])
    val = structs[structs.split != 'train'].set_index('ik14')
    pf = pq.ParquetFile(os.path.join(ROOT, 'train.parquet'))
    parts = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=['ingest_lib', 'inchikey14', 'precursor_error_ppm'] + TEST_COLS).to_pandas()
        t = t[t.inchikey14.isin(val.index) & t.adduct.isin(TEST_ADDUCTS) & (t.precursor_error_ppm.abs() <= 20)]
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df['split'] = df.inchikey14.map(val.split)
    rng = np.random.default_rng(0)
    for name in ['val_np', 'val_rand']:
        d = df[df.split == name]
        if name == 'val_np':
            d = d[d.ingest_lib == 'enveda-np-examples']
        rows = []
        for ik, g in d.groupby('inchikey14'):
            if name == 'val_rand':                       # queries come from a single library, like a real acquisition
                lib = g.ingest_lib.value_counts().index[0]
                g = g[g.ingest_lib == lib]
            if len(g) > max_spectra:
                g = g.iloc[rng.choice(len(g), max_spectra, replace=False)]
            rows.append(g)
        q = pd.concat(rows, ignore_index=True)
        q['molecule_id'] = 'v_' + q.inchikey14
        q['spectrum_id'] = [f's_{i:06d}' for i in range(len(q))]
        truth = q.groupby('molecule_id').agg(ik14=('inchikey14', 'first'), query_lib=('ingest_lib', 'first')).reset_index()
        truth['smiles'] = truth.ik14.map(val.smiles)
        q[['molecule_id', 'spectrum_id'] + TEST_COLS].to_parquet(os.path.join(WORK, f'{name}.parquet'), index=False)
        truth[['molecule_id', 'smiles', 'ik14', 'query_lib']].to_csv(os.path.join(WORK, f'{name}_truth.csv'), index=False)
        print(name, len(q), 'spectra', q.molecule_id.nunique(), 'molecules', '| query libs:', truth.query_lib.value_counts().to_dict())


if __name__ == '__main__':
    main()
