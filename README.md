# Enveda CASMI 2026 — Molecule ID From Mass Spectra

[![Competition](https://img.shields.io/badge/Kaggle-Enveda%20CASMI%202026-blue)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
[![Leaderboard](https://img.shields.io/badge/Public%20LB-0.207%20(V1)-green)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)
[![Metric](https://img.shields.io/badge/Metric-MRR%4025%20(InChIKey14)-orange)]()
[![Hardware](https://img.shields.io/badge/Compute-NVIDIA%20RTX%204090-purple)]()

Predict 2D chemical structures (SMILES) of small molecules and natural products directly from tandem mass spectra (LC-MS/MS). Built for the **Critical Assessment of Small Molecule Identification (CASMI) 2026** challenge.

---

##  Pipeline Architecture

For every query spectrum, the pipeline calculates the adduct-aware neutral mass ($\pm 10\text{ ppm}$, calibrated for timsTOF instrument bias) and merges candidates across three complementary tiers:

```mermaid
flowchart TD
    Q["Query LC-MS/MS Spectrum<br/>(m/z, intensity, adduct, CE)"] --> C["Adduct & Instrument Calibration<br/>(+2.0 ppm pos / +0.6 ppm neg)"]
    C --> MW["Mass Window Filter (±10 ppm)"]
    
    MW --> T1["Tier 1: Spectral Library Search<br/>(Cosine similarity over 2M training spectra)"]
    MW --> T2["Tier 2: Offline Databases<br/>(COCONUT: 461k | PubChem: 84.2M)"]
    MW --> T3["Tier 3: De Novo Transformer<br/>(Spectrum → SMILES with KV-Cache)"]
    
    T1 --> M["Candidate Pool Merger & Deduplication"]
    T2 --> M
    T3 --> M
    
    M --> F["Feature Extraction<br/>(FP cosine, seq likelihood, peak counts, DB source)"]
    F --> R["Listwise MLP Ranker<br/>(Trained on held-out validation spectra)"]
    R --> S["Top-25 Unique InChIKey14 Predictions<br/>(submission.csv)"]
```

### Key Components

1. **Adduct Arithmetic & Instrument Calibration ([`casmi/common.py`](casmi/common.py))**:
   - Accurately derives neutral mass for all 10 competition adducts (`[M+H]+`, `[M+Na]+`, `[M-H]-`, etc.).
   - Corrects for systematic timsTOF spectrometer bias (+2.0 ppm in positive mode, +0.6 ppm in negative mode).

2. **Spectral Library Search ([`casmi/libsearch.py`](casmi/libsearch.py))**:
   - Searches 2.04 million reference spectra in `train.parquet`.
   - Delivers **~0.90–0.95 MRR** when the compound or an isomer has public reference spectra (Class 1).

3. **Offline Candidate Retrieval ([`casmi/candidates.py`](casmi/candidates.py))**:
   - **COCONUT Database**: 461,646 natural product structures.
   - **PubChem Database**: 84,259,636 filtered, neutral, single-component small molecules.
   - Sub-15ms queries via binary-searched Parquet row-group statistics without loading full tables into RAM.

4. **Spectrum-to-Structure Transformer ([`casmi/model.py`](casmi/model.py), [`casmi/train_v2.py`](casmi/train_v2.py))**:
   - **SpectrumEncoder**: Sinusoidal m/z embeddings + peak MLPs + precursor/adduct conditioning token.
   - **Morgan Fingerprint Head**: 4096-bit predicted fingerprint for ranking database candidates.
   - **SmilesDecoder**: Causal autoregressive decoder with KV-cache for candidate log-likelihood and de novo generation.
   - **V2 Generalization Fixes**:
     - 4 randomized SMILES variants per structure to eliminate string memorization.
     - 25% batch mix of structure-only COCONUT examples to learn natural product chemistry from mass alone.
     - Balanced sampling capping spectra per structure to at most 6 per epoch.
     - Dropout 0.2 with 74-token vocabulary.

5. **Candidate Re-Ranker ([`casmi/ranker_lab.py`](casmi/ranker_lab.py), [`casmi/pipeline.py`](casmi/pipeline.py))**:
   - Listwise neural ranker trained with L-BFGS to order candidates from all three tiers.
   - Deduplicates candidates by tautomer-canonical InChIKey first block (InChIKey14).

---

## 📂 Repository Structure

```text
CASMI/
├── README.md                      # Project documentation and quickstart
├── sample_submission.csv          # Competition submission format template
├── test.parquet                   # Test spectra (400 evaluation molecules)
├── train.parquet                  # Training spectra (2.04M spectra, 275k structures)
│
├── casmi/                         # Core Python package (runs locally & on Kaggle)
│   ├── common.py                  # Adduct arithmetic, metric keys, tokenizers, Morgan FPs
│   ├── metric.py                  # MRR@25 replica matching Kaggle evaluation
│   ├── candidates.py              # CandidateDB fast mass-window Parquet query engine
│   ├── libsearch.py               # Spectral library search over train.parquet
│   ├── model.py                   # Transformer architecture (Encoder + FP Head + Decoder)
│   ├── infer.py                   # ModelScorer: fingerprints, candidate scoring, de novo sampling
│   ├── pipeline.py                # End-to-end candidate merging, feature extraction, ranking
│   ├── train.py                   # V1 model training loop
│   ├── train_v2.py                # V2 model training loop (randomized SMILES + COCONUT pretrain)
│   ├── dump_features.py           # Dumps candidate feature tables for offline ranker training
│   ├── ranker_lab.py              # Cross-validated ranker training (L-BFGS listwise loss)
│   └── run_test.py                # Pipeline execution entrypoint (generates submission.csv)
│
├── docs/                          # Detailed guides and background documentation
│   ├── PROGRESS.md                # Progress log, resume guide, and experimental findings
│   ├── Enveda_CASMI_2026_*.md     # Kaggle competition overview and dataset specifications
│   └── CASMI_denovo_tutorial_*.md # Initial tutorial references and notes
│
├── notebooks/                     # Jupyter notebooks
│   ├── casmi_v1_kaggle.ipynb      # Standalone self-contained V1 Kaggle submission notebook
│   └── casmi-denovo-tutorial-*.ipynb # Initial exploratory baseline notebook
│
├── kaggle_assets/                 # Files packaged for Kaggle private dataset
│   ├── model.pt                   # Transformer weights
│   ├── vocab.json                 # SMILES token vocabulary
│   ├── ranker.json                # Fitted candidate ranker weights & scalers
│   ├── structures.parquet         # Training structures and InChIKey index
│   ├── coconut.parquet            # 461k COCONUT candidates
│   ├── pubchem.parquet            # 84.2M PubChem candidates
│   └── rdkit-*.whl                # Offline RDKit wheel for Kaggle runtime
│
├── scripts/                       # Automation and utility scripts
│   ├── build_kaggle_notebook.py   # Assembles self-contained notebook from casmi/ modules
│   ├── package_assets.py          # Packages kaggle_assets/ and rebuilds notebook
│   ├── build_pubchem_hf.py        # Streams PubChem dump, filters, and generates Parquet DB
│   └── legacy/                    # Previous prototype scripts (casmi_v1.py, etc.)
│
├── work/                          # Training outputs, checkpoints, and offline DBs (ignored)
│   ├── ckpt/                      # Model checkpoints (model_v2_best.pt, model_ep*.pt)
│   ├── db/                        # Generated Parquet DBs (coconut.parquet, pubchem.parquet)
│   └── v2/                        # V2 tokenized datasets and vocabulary
│
└── output/                        # Local submission predictions
    └── submission.csv             # Verified 400-molecule test prediction
```

---

## 🚀 Quickstart & Usage

### 1. Environments
* **GPU Training**: `~/casmi-gpu-venv/bin/python` (PyTorch 2.10 + CUDA, bf16 enabled, RDKit).
* **CPU / Analysis**: `~/miniconda3/envs/casmi/bin/python` (RDKit, pandas, pyarrow, scikit-learn).

### 2. Train V2 Model
To train the generalized spectrum-to-structure model on the RTX 4090:
```bash
~/casmi-gpu-venv/bin/python -m casmi.train_v2 --epochs 8 --batch-size 256
```

### 3. Generate Local Submission
Run inference on `test.parquet`:
```bash
~/miniconda3/envs/casmi/bin/python -m casmi.run_test \
    --test test.parquet \
    --train train.parquet \
    --assets kaggle_assets \
    --out output/submission.csv
```

### 4. Package Assets & Rebuild Kaggle Notebook
```bash
~/miniconda3/envs/casmi/bin/python scripts/package_assets.py --model work/ckpt/model_v2_best.pt
```

### 5. Submit to Kaggle
1. Upload `kaggle_assets/` as a private Kaggle dataset (`casmi-v1-assets`).
2. Open [`notebooks/casmi_v1_kaggle.ipynb`](notebooks/casmi_v1_kaggle.ipynb) in Kaggle.
3. Attach the competition data and your `casmi-v1-assets` dataset.
4. Select **GPU T4**, toggle **Internet Off**, and click **Save Version → Save & Run All (Commit)**.
5. In the Output tab, click **Submit to Competition**.

---

## 📊 Experimental Results

### Held-Out Validation (Natural Products on timsTOF)

| Scenario / Class | Definition | V1 Model (COCONUT only) | V2 Expected (PubChem + Un-memorized) |
|---|---|---|---|
| **Class 1 (Library)** | Public spectrum exists in library | **~0.90 MRR** | **~0.92 MRR** |
| **Class 2 (Database)** | No spectrum, but structure in DB | **~0.53 MRR** | **~0.60+ MRR** (84M PubChem coverage) |
| **Class 3 (De Novo)** | Novel structure, generation only | **~0.18 MRR** | **~0.25+ MRR** (randomized SMILES) |

* **Kaggle Public Leaderboard (V1)**: **0.207** (using only COCONUT; confirmed 100% compliant and offline-compatible).
* **Target (V2)**: **0.35–0.45+** by incorporating full PubChem candidate retrieval and the V2 generalized model.
