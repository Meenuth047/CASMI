"""Assemble kaggle_assets/ (the Kaggle dataset the notebook reads) and rebuild the notebook.

    python scripts/package_assets.py [--model work/ckpt/model_epN.pt]
"""
import os
import sys
import json
import shutil
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK, OUT = os.path.join(ROOT, 'work'), os.path.join(ROOT, 'kaggle_assets')

ap = argparse.ArgumentParser()
default_model = os.path.join(WORK, 'ckpt', 'model_v2_best.pt')
if not os.path.exists(default_model):
    default_model = os.path.join(WORK, 'ckpt', 'model.pt')
ap.add_argument('--model', default=default_model)
ap.add_argument('--slug', default='casmi-v1-assets')
args = ap.parse_args()

vocab_path = os.path.join(WORK, 'v2', 'vocab.json')
if not os.path.exists(vocab_path):
    vocab_path = os.path.join(WORK, 'vocab.json')

os.makedirs(OUT, exist_ok=True)
files = {
    args.model: 'model.pt',
    vocab_path: 'vocab.json',
    os.path.join(WORK, 'ranker.json'): 'ranker.json',
    os.path.join(WORK, 'structures.parquet'): 'structures.parquet',
    os.path.join(WORK, 'db', 'coconut.parquet'): 'coconut.parquet',
    os.path.join(WORK, 'db', 'pubchem.parquet'): 'pubchem.parquet',
}
for src, name in files.items():
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, name))
        print(f'  {name:22s} {os.path.getsize(src) / 1e6:9.1f} MB')
    else:
        print(f'  {name:22s} MISSING ({src})' + ('  <- optional' if name == 'pubchem.parquet' else ''))

user = 'thakurmeenukumari'
try:
    with open(os.path.expanduser('~/.kaggle/kaggle.json')) as fh:
        user = json.load(fh).get('username', user)
except Exception:
    pass
meta = {'title': 'CASMI V1 assets', 'id': f'{user}/{args.slug}', 'licenses': [{'name': 'CC-BY-NC-SA-4.0'}]}
with open(os.path.join(OUT, 'dataset-metadata.json'), 'w') as fh:
    json.dump(meta, fh, indent=1)
print('dataset id:', meta['id'])
subprocess.check_call([sys.executable, os.path.join(ROOT, 'scripts', 'build_kaggle_notebook.py'), '--version', 'v3'])
