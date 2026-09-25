"""Generate casmi_v1_kaggle.ipynb: a self-contained inference notebook (code embedded, weights/DBs from a Kaggle dataset)."""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ['common.py', 'model.py', 'infer.py', 'libsearch.py', 'candidates.py', 'pipeline.py', 'run_test.py']


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text}


cells = [md(
    '# CASMI 2026 — V1: library search + database retrieval + de novo generation\n\n'
    'For every unknown molecule the adduct-aware neutral mass (±10 ppm, instrument bias calibrated) defines a candidate set from three sources:\n\n'
    '1. **training structures** ranked by spectral similarity to their public library spectra (class 1),\n'
    '2. **COCONUT + PubChem** structures in the mass window (class 2),\n'
    '3. **de novo samples** from a spectrum→SMILES transformer, kept only if their mass matches (class 3).\n\n'
    'One transformer encoder feeds a Morgan-fingerprint head (ranks database candidates) and a SMILES decoder (candidate likelihood + generation). '
    'A small linear ranker fitted on held-out molecules orders the merged candidates; the top 25 unique tautomer-canonical InChIKey blocks are submitted.\n\n'
    '**Inputs:** the competition data, plus the dataset holding `model.pt`, `vocab.json`, `ranker.json`, `structures.parquet`, `coconut.parquet`, `pubchem.parquet` (all built from public data: the competition training set, COCONUT (CC0) and PubChem).'
)]

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
out = os.path.join(ROOT, 'casmi_v1_kaggle.ipynb')
with open(out, 'w') as fh:
    json.dump(nb, fh, indent=1)
print('wrote', out, f'({len(cells)} cells)')
