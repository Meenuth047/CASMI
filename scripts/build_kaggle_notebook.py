"""Generate a self-contained Kaggle inference notebook (code embedded, weights/DBs from a Kaggle dataset)."""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ['common.py', 'model.py', 'infer.py', 'libsearch.py', 'candidates.py', 'pipeline.py', 'run_test.py']

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v3', help='Pipeline version (e.g. v1, v2, v3)')
parser.add_argument('--out', type=str, default=None, help='Output path for the notebook')
args = parser.parse_args()

v_num = args.version.lower().lstrip('v')
title_ver = f'V{v_num}'

desc = f'''# CASMI 2026 — {title_ver} Pipeline: 3-Tier Spectral Retrieval + Generative Transformer + Multi-Task Ranker

For every unknown molecule, the adduct-aware neutral mass (±10 ppm, instrument bias calibrated) defines a candidate set from three sources:
1. **Spectral Library Search**: Training structures ranked by cosine spectral similarity (Class 1).
2. **Database Mass-Window Retrieval**: 84.2M PubChem + 461k COCONUT structures (Class 2) with expanded PubChem shortlist (max 350).
3. **De Novo Generation**: Transformer sampling (128 samples/spectrum at T=1.0) with mass-matching filtering (Class 3).

Candidates are scored via:
- Model predicted Morgan fingerprints (cosine similarity to spectrum embedding)
- Autoregressive sequence log-likelihood & token-normalized scores
- 5-Fold MLP Ranker trained on 1,079 molecules (val_np + val_rand) with library margin and confidence features.
Outputs top-25 deduplicated tautomer-canonical InChIKey14 SMILES.
'''

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text}


cells = [md(desc)]

cells.append(code('''import os, sys, glob, subprocess

COMP_DIR = next(iter(glob.glob('/kaggle/input/**/enveda-CASMI26-molecule-id-mass-spectra', recursive=True)), '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra')
hits = glob.glob('/kaggle/input/**/model.pt', recursive=True)
assert hits, 'Attach the CASMI V1 assets dataset (model.pt, vocab.json, structures.parquet, coconut.parquet, ...) to this notebook'
ASSETS_DIR = os.path.dirname(hits[0])
print('competition data:', COMP_DIR, os.listdir(COMP_DIR))
print('assets          :', ASSETS_DIR, os.listdir(ASSETS_DIR))

# RDKit: use the environment's copy if present (Dependency Manager: rdkit==2026.3.3), else install the bundled wheel offline.
try:
    import rdkit
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--find-links', ASSETS_DIR, 'rdkit'])
    import rdkit
print('rdkit', rdkit.__version__)
os.makedirs('/kaggle/working/casmi', exist_ok=True)
open('/kaggle/working/casmi/__init__.py', 'w').close()
os.chdir('/kaggle/working')
sys.path.insert(0, '/kaggle/working')'''))

for name in MODULES:
    with open(os.path.join(ROOT, 'casmi', name)) as fh:
        cells.append(code(f'%%writefile casmi/{name}\n' + fh.read()))

cells.append(code('''from casmi.run_test import run

submission = run(
    test_path=os.path.join(COMP_DIR, 'test.parquet'),
    sample_submission_path=os.path.join(COMP_DIR, 'sample_submission.csv'),
    train_path=os.path.join(COMP_DIR, 'train.parquet'),
    assets_dir=ASSETS_DIR,
    out_path='/kaggle/working/submission.csv',
    n_proc=4,
)
submission.head()'''))

nb = {
    'cells': [{**c, 'source': c['source'].splitlines(keepends=True)} for c in cells],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'kaggle': {'accelerator': 'nvidiaTeslaT4', 'dataSources': [], 'isInternetEnabled': False, 'language': 'python',
                   'sourceType': 'notebook', 'isGpuEnabled': True},
    },
    'nbformat': 4, 'nbformat_minor': 4,
}
os.makedirs(os.path.join(ROOT, 'notebooks'), exist_ok=True)
out = args.out or os.path.join(ROOT, 'notebooks', f'casmi_{args.version.lower()}_kaggle.ipynb')
with open(out, 'w') as fh:
    json.dump(nb, fh, indent=1)
print('wrote', out, f'({len(cells)} cells)')
