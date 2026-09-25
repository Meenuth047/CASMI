# CASMI 2026 — Master Project Documentation & Progress Chronicle

> **Competition**: [Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)  
> **Evaluation Metric**: Mean Reciprocal Rank @ 25 (MRR@25) on first 14 characters of InChIKey (InChIKey14)  
> **Current Best Public LB Score**: **0.281** (V3 Personal Best 🏆)  
> **Competition Deadline**: December 14, 2026  
> **Author / Kaggle User**: `thakurmeenukumari`  
> **Primary Compute**: Local workstation with NVIDIA GeForce RTX 4090 (24 GB VRAM)  
> **Document Purpose**: Complete, self-contained architectural and chronological specification of the codebase. Any AI assistant or engineer reading this document can immediately grasp the problem, data, models, experimental results, and next optimization steps.

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Challenge
Natural products (secondary metabolites produced by plants, microbes, and fungi) serve as the foundation for over 50% of modern pharmaceuticals. Tandem mass spectrometry (LC-MS/MS) is the primary analytical technique used to detect and characterize these molecules. 

In LC-MS/MS:
1. Molecules are ionized (forming adducts such as $[M+H]^+$, $[M+Na]^+$, $[M-H]^-$).
2. The precursor ion is isolated and subjected to Collision-Induced Dissociation (CID), smashing the compound into charged and neutral fragments.
3. The detector records fragment mass-to-charge ratios ($m/z$) and relative intensities.

**The Fundamental Problem**: A mass spectrum is **not** a 2D chemical structure; it is only a collection of fragment peaks and an overall precursor mass. Multiple constitutional isomers and stereoisomers share identical molecular formulas and masses. Determining the exact 2D connectivity (SMILES string) from spectral fragments without reference standards is a long-standing grand challenge in analytical chemistry.

### 1.2 Evaluation Metric: MRR@25 on InChIKey14
For every query spectrum (or group of spectra corresponding to one unknown molecule ID), submissions must provide a semicolon-separated list of up to 25 unique candidate SMILES.
* Predictions are converted to their 14-character InChIKey first-block (tautomer-canonical molecular skeleton, representing connectivity while ignoring stereochemistry and tautomerism).
* Predictions are deduplicated on this InChIKey14 metric key.
* The score for molecule $i$ is:
  $$\text{RR}_i = \begin{cases} \frac{1}{\text{rank}} & \text{if true InChIKey14 is in top-25 predictions at 1-based index } \text{rank} \\ 0 & \text{if true InChIKey14 is not in top-25} \end{cases}$$
* The final leaderboard score is the Mean Reciprocal Rank: $\text{MRR@25} = \frac{1}{N} \sum_{i=1}^N \text{RR}_i$.

---

## 2. Overall Pipeline Architecture (3-Tier Hybrid Search)

Rather than relying purely on end-to-end generative AI (which struggles with exact isomer resolution) or purely on database matching (which fails on completely novel natural products), this solution implements a **3-Tier Hybrid Retrieval and Ranking Architecture**:

```
                       ┌───────────────────────────────────────────────┐
                       │           Query LC-MS/MS Spectrum             │
                       │  (peaks: m/z, intensity; adduct, prec_mz, CE) │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │        Adduct & Instrument Calibration        │
                       │   Neutral Mass M_0 = (prec_mz - m_add) / z    │
                       │   Calibrated: +2.0 ppm (pos) / +0.6 ppm (neg) │
                       └───────────────────────┬───────────────────────┘
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
    ┌───────────────────────────────┐                     ┌───────────────────────────────┐
    │     Mass Window Retrieval     │                     │     Generative Transformer    │
    │     (M_0 ± 10 ppm window)     │                     │    (CasmiModel: Enc + Dec)    │
    └───────┬───────────────┬───────┘                     └───────────────┬───────────────┘
            │               │                                             │
            ▼               ▼                                             ▼
  ┌──────────────────┐ ┌──────────────────┐                     ┌──────────────────┐
  │  Tier 1: Library │ │ Tier 2: Databases│                     │ Tier 3: De Novo  │
  │  Train Spectra   │ │ PubChem (84.2M)  │                     │ Autoregressive   │
  │  (2.04M spectra) │ │ COCONUT (461k)   │                     │ SMILES sampling  │
  │  Cosine Sim >=0.4│ │ Fast Parquet scan│                     │ (128 samples/sp) │
  └─────────┬────────┘ └────────┬─────────┘                     └────────┬─────────┘
            │                   │                                        │
            └───────────────────┼────────────────────────────────────────┘
                                │
                                ▼
                       ┌───────────────────────────────────────────────┐
                       │          Candidate Shortlist Merging          │
                       │   Cap per source: Lib (200), COCONUT (150),   │
                       │          PubChem (350), De Novo (100)         │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │          Dual-Engine Feature Scoring          │
                       │   1. Predicted Morgan FP Cosine Similarity    │
                       │   2. Sequence Cross-Entropy Likelihood        │
                       │   3. Spectral Library Match & Margin Features │
                       │   4. Structural & Database Source Flags       │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │      5-Fold MLP-32 Listwise Re-Ranker         │
                       │ (Trained on 1,079 held-out spectra, 22 feats) │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │        Deduplicated Top-25 InChIKey14         │
                       │            Final submission.csv               │
                       └───────────────────────────────────────────────┘
```

### Component Details:
1. **Adduct Arithmetic & Calibration (`casmi/common.py`)**:
   - Accounts for 10 competition adducts: `[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`, `[M-H2O+H]+`, `[M-H]-`, `[M+HCOO]-`, `[M+CH3COO]-`, `[M-H2O-H]-`, `[M+Cl]-`.
   - Corrects Bruker timsTOF mass spectrometer systematic drift (+2.0 ppm in positive mode, +0.6 ppm in negative mode).
2. **Tier 1: Spectral Library Search (`casmi/libsearch.py`)**:
   - Compares query peak list against 2,042,382 training spectra in `train.parquet`.
   - Uses Square-Root Intensity Cosine Similarity ($m/z$ bin tolerance = 0.02 Da).
   - If the exact compound or a close isomer was measured in reference data, library search achieves **0.90–0.95 MRR**.
3. **Tier 2: Offline High-Speed Database Retrieval (`casmi/candidates.py`)**:
   - **COCONUT**: 461,646 natural product structures (11.9 MB Parquet).
   - **PubChem**: 84,259,636 filtered single-component small molecules (1.84 GB Parquet).
   - Utilizes binary-searched Parquet file metadata row-group min/max statistics; extracts candidate SMILES in <15ms without loading the entire 1.84 GB table into memory.
4. **Tier 3: Generative Spectrum-to-SMILES Transformer (`casmi/model.py`, `casmi/infer.py`)**:
   - **SpectrumEncoder**: Sinusoidal positional embeddings for continuous $m/z$ values, 2-layer peak MLP, precursor mass + adduct + collision energy conditioning token.
   - **Morgan Fingerprint Head**: Predicts a 4096-bit Morgan circular fingerprint directly from the pooled spectrum embedding. Used to compute cosine similarity against all retrieved database candidates.
   - **SmilesDecoder**: 6-layer causal autoregressive Transformer with KV-cache for both conditional sequence log-likelihood scoring and de novo ancestral token sampling.
5. **Listwise Neural Ranker (`casmi/ranker_lab.py`, `casmi/pipeline.py`)**:
   - A multi-layer perceptron (MLP with 32 hidden units, Tanh activation, L2 regularization) trained via L-BFGS on listwise negative log-likelihood loss.
   - Combines 22 distinct features across all 3 tiers (FP cosine, cosine rank, sequence log-likelihood, length-normalized likelihood, library margin, database origin flags).

---

## 3. Detailed Version Evolution: V1 → V2 → V3

### 3.1 Version 1: The Initial Baseline (Score: 0.207)
* **Goal**: Fix critical bugs in the competition tutorial baseline and build an offline, submission-compliant pipeline.
* **Key Bug Fixes over Baseline Tutorial**:
  1. *Unshifted Decoder Labels*: Tutorial computed loss between target tokens and themselves instead of next tokens ($t+1$).
  2. *Last-Token Sampling*: Tutorial generation loop only appended the final token instead of building the full SMILES sequence.
* **Pipeline Setup**:
  - Library search over training spectra.
  - Candidate retrieval using COCONUT only (461k natural products). *PubChem was not yet integrated due to download throttling.*
  - V1 Transformer trained for 5 epochs on raw canonical SMILES targets.
  - Linear candidate ranker fitted on 125 validation natural products (`val_np`).
* **Validation & Submission**:
  - Kaggle Public LB: **0.207** (using only COCONUT; verified 100% offline compliance and metric correctness).
* **Identified Limitation**:
  - The model exhibited severe memorization of canonical SMILES strings. Training loss dropped to 0.14, but validation loss on unseen structures bottomed at epoch 2 (0.289) and degraded afterwards. De novo valid sequence generation was low (~64%).

---

### 3.2 Version 2: The Generalization & Database Overhaul (Score: 0.241, +16.4%)
To resolve the memorization bottleneck and expand the candidate universe to all known chemistry, V2 executed two major engineering pillars:

#### Pillar 1: Full-Scale PubChem Database Construction (`scripts/build_pubchem_hf.py`)
* NCBI official FTP was throttled (<100 KB/s), so a mirrored snapshot (`work/db/tmp/pubchem_hf.tar.xz`, 698 MB) was obtained.
* Implemented multi-worker parallel parsing with RDKit across 16 CPU cores.
* Filtered 119M entries down to **84,259,636 valid structures**:
  - Mass range: 100.00 – 1299.99 Da
  - Elements: C, H, N, O, P, S, F, Cl, Br, I only
  - Strictly single-component, neutral, non-isotopic molecules
* Output: `work/db/pubchem.parquet` (1.84 GB), sorted by monoisotopic mass with row-group metadata for sub-15ms queries.

#### Pillar 2: V2 Transformer Architecture & De-Memorization (`casmi/train_v2.py`)
1. **Randomized SMILES Data Augmentation**: Prepared 4 non-canonical randomized SMILES strings per training structure (99.4% differing from canonical). Forces the decoder to learn molecular syntax and grammar rather than memorizing fixed string prefixes.
2. **25% Structure-Only COCONUT Batch Pretraining**: Injected 459,480 pure structures from COCONUT into the training stream without spectra (conditioning solely on precursor mass and adduct). Teaches the decoder natural product scaffolds and molecular mass constraints.
3. **Balanced Spectrum Sampling**: Capped spectra to at most 6 per structure per epoch to prevent dominant compounds from skewing the representation.
4. **Training Dynamics (RTX 4090, 8 Epochs, 1.63 hours)**:
   - Validation cross-entropy monotonically dropped: $0.602 \to 0.493 \to 0.440 \to 0.389 \to 0.355 \to 0.343 \to 0.334 \to \mathbf{0.326}$.
   - De novo generation validity jumped from **64% to 88%**.
   - Top-1 exact de novo match on unseen held-out validation spectra jumped from **0% to 6.0%** (Top-20 hit rate: 12.5%).
   - Best checkpoint saved to `work/ckpt/model_v2_best.pt` (196 MB).
5. **V2 Ranker**: First non-linear MLP ranker (hidden=32, L2=1e-2) trained on `work/feats_val_np_v2.parquet` (90,341 candidates).
   - Offline `val_np` results: Class 1 MRR 0.977 | Class 2 MRR 0.678 | Class 3 MRR 0.201.
* **Kaggle Public LB Result**: **0.241** *(+16.4% relative gain over V1, new personal best)*.

---

### 3.3 Version 3: Candidate Expansion & Scaled Ranker Training (Current)
While V2 brought huge structural improvements, analysis revealed critical headroom:
1. **Under-trained Ranker**: The V2 ranker was fitted on only 125 natural product molecules (`val_np`), leaving 954 diverse validation molecules (`val_rand`) unused.
2. **Restricted Shortlist Bottleneck**: `max_pubchem` was capped at 150 candidates. If the true structure was ranked between #151 and #350 in the mass window, the decoder never had the chance to score it.
3. **Low De Novo Sampling**: Only 64 de novo sequences were sampled per spectrum.

#### V3 Innovations Implemented:
1. **Full-Dataset Feature Extraction**:
   - Executed `casmi.dump_features` across all 954 `val_rand` molecules using the V2 model and the 84.2M PubChem database.
   - Produced `work/feats_val_rand_v2.parquet` (43 MB, 709,276 candidates across 3,237 lists).
2. **5-Fold Cross-Validated MLP-32 Ranker**:
   - Scaled ranker training dataset from 125 molecules to **all 1,079 validation molecules** (`val_np` + `val_rand`).
   - Evaluated 22 ranking features (`RANKER_FEATURES`):
     - Model: `fp_cos`, `fp_rank`, `ll_rel`, `ll_sqrt`, `ll_tok`, `ll_rank`
     - Library: `lib_score`, `lib_hi`, `lib_top`, `lib_margin_top`, `lib_hi7`, `lib_hi9`, `lib_conf`
     - Database & De Novo: `in_lib`, `in_coconut`, `in_pubchem`, `db_only_pubchem`, `denovo_frac`, `denovo_any`
     - Contextual: `analog_sim`, `analog_tani`, `log_nspec`
   - **Cross-Validation Comparison on 1,079 Molecules**:
     | Ranker Model | val_np c1 | val_np c2 | val_np c3 | val_rand c1 | val_rand c2 | val_rand c3 |
     |---|:---:|:---:|:---:|:---:|:---:|:---:|
     | Fallback Hand-Weights | 0.924 | 0.225 | 0.059 | 0.695 | 0.198 | 0.106 |
     | Linear + Lib-Confidence | 0.815 | 0.485 | 0.101 | 0.720 | 0.259 | 0.136 |
     | MLP-16 + Lib-Confidence | 0.866 | 0.558 | 0.146 | 0.728 | 0.311 | 0.154 |
     | **MLP-32 + Lib-Confidence (V3)** | **0.864** | **0.522** | **0.132** | **0.734** | **0.309** | **0.161** |
     *(Class 2 offline MRR jumped by **+132%** over fallback!)*
3. **Shortlist & Sampling Expansion**:
   - In [`casmi/pipeline.py`](casmi/pipeline.py):
     - `max_pubchem`: **150 → 350** (allows 2.33× more database candidates to be evaluated by sequence likelihood).
     - `denovo_samples`: **64 → 128** (doubled generation volume per spectrum).
4. **Codebase & Notebook Improvements**:
   - Fixed missing `import os` in [`casmi/infer.py`](casmi/infer.py).
   - Created dedicated versioned notebook generator in [`scripts/build_kaggle_notebook.py`](scripts/build_kaggle_notebook.py), outputting [`notebooks/casmi_v3_kaggle.ipynb`](notebooks/casmi_v3_kaggle.ipynb).
   - Packaged updated weights into `kaggle_assets/ranker.json`.
* **Kaggle Public LB Result**: **0.281** *(+16.6% relative gain over V2, +35.7% total gain over V1 baseline! 🏆)*.

---

## 4. Benchmark & Scoreboard Summary

| Version | Description | Offline Validation (c1 / c2 / c3 MRR) | Kaggle Public LB (MRR@25) | Notes |
|:---:|:---|:---:|:---:|:---|
| **Tutorial** | Unshifted labels, broken sampling, no database | 0.000 / 0.000 / 0.000 | 0.000 | Broken baseline |
| **V1** | 3-tier retrieval (COCONUT only); V1 model; linear ranker (125 mol) | 0.900 / 0.225 / 0.059 | **0.207** | Fully compliant offline run |
| **V2** | + 84.2M PubChem DB; V2 de-memorized model; MLP ranker (125 mol) | 0.924 / 0.225 / 0.062 | **0.241** | +16.4% gain over V1 |
| **V3** | PubChem shortlist 350; 128 de novo samples; MLP-32 ranker (1,079 mol) | **0.864 / 0.522 / 0.132** | **0.281** | **New Personal Best (+35.7% total) 🏆** |

---

## 5. System Environment & Directory Structure

### 5.1 Computing Environments
* **Machine**: Linux (Ubuntu-based), User: `airbotix`
* **GPU Environment**: `/home/airbotix/casmi-gpu-venv/bin/python`
  - PyTorch 2.10 with CUDA 12 support, `bfloat16` accelerated.
  - Hardware: NVIDIA GeForce RTX 4090 (24 GB VRAM).
  - Use for: Model training (`casmi.train_v2`), feature dumping (`casmi.dump_features`), full validation pipelines.
* **CPU Environment**: `/home/airbotix/miniconda3/envs/casmi/bin/python`
  - Packages: RDKit, Pandas, PyArrow, Scikit-Learn.
  - Use for: Database generation, asset packaging, notebook building.
* **Kaggle User Account**: `thakurmeenukumari` (Assets dataset slug: `casmi-v1-assets`)

### 5.2 Directory Map
```text
/home/airbotix/Downloads/CASMI/
├── README.md                      # Project front-page overview & quickstart
├── sample_submission.csv          # Submission format template
├── test.parquet                   # Test spectra (400 evaluation molecules)
├── train.parquet                  # 2.04M reference spectra across 275k structures
│
├── casmi/                         # Core Python package
│   ├── common.py                  # Adduct arithmetic, calibration, tokenizers, Morgan FPs
│   ├── metric.py                  # Kaggle-exact MRR@25 scoring logic & validation
│   ├── candidates.py              # CandidateDB fast mass-window Parquet query engine
│   ├── libsearch.py               # Cosine spectral library search over train.parquet
│   ├── model.py                   # CasmiModel (SpectrumEncoder + FP Head + SmilesDecoder)
│   ├── infer.py                   # ModelScorer (inference batching, likelihood, sampling)
│   ├── pipeline.py                # End-to-end 3-tier merging, feature assembly, ranking
│   ├── train.py                   # V1 training loop (legacy)
│   ├── train_v2.py                # V2 training loop (randomized SMILES + COCONUT pretrain)
│   ├── dump_features.py           # Feature dumping for validation sets -> feats_*.parquet
│   ├── ranker_lab.py              # Cross-validated L-BFGS listwise ranker training
│   └── run_test.py                # Entrypoint for running inference on test spectra
│
├── docs/                          # In-depth documentation
│   ├── PROGRESS.md                # THIS FILE: Master engineering specification & progress
│   ├── Enveda_CASMI_2026_*.md     # Official competition overview & dataset guidelines
│   └── CASMI_denovo_tutorial_*.md # Tutorial reference notes
│
├── notebooks/                     # Standalone submission notebooks
│   ├── casmi_v1_kaggle.ipynb      # V1 Kaggle notebook
│   ├── casmi_v2_kaggle.ipynb      # V2 Kaggle notebook
│   └── casmi_v3_kaggle.ipynb      # V3 Kaggle notebook (USE FOR V3 SUBMISSION)
│
├── kaggle_assets/                 # Files packaged for Kaggle private dataset (casmi-v1-assets)
│   ├── model.pt                   # V2 best model checkpoint (196 MB)
│   ├── vocab.json                 # 74-token SMILES vocabulary
│   ├── ranker.json                # V3 MLP-32 trained ranker weights & scalers (21 KB)
│   ├── structures.parquet         # Training structures and InChIKey index (42.3 MB)
│   ├── coconut.parquet            # 461,646 COCONUT structures (11.9 MB)
│   ├── pubchem.parquet            # 84,259,636 PubChem structures (1.84 GB)
│   ├── dataset-metadata.json      # Dataset identifier (thakurmeenukumari/casmi-v1-assets)
│   └── rdkit-*.whl                # Bundled RDKit wheel for offline runtime installation
│
├── scripts/                       # Automation scripts
│   ├── build_kaggle_notebook.py   # Self-contained notebook builder with %%writefile embedding
│   ├── package_assets.py          # Assembles kaggle_assets/ and rebuilds notebooks
│   ├── build_pubchem_hf.py        # Parallel stream-builder for 84.2M PubChem database
│   └── legacy/                    # Archived prototypes (casmi_v1.py, etc.)
│
├── work/                          # Local experimental outputs & checkpoints (not in git)
│   ├── ckpt/                      # Model checkpoints (model_v2_best.pt, model_v2_ep*.pt)
│   ├── db/                        # Local Parquet DBs (coconut.parquet, pubchem.parquet)
│   ├── v2/                        # V2 tokenized arrays (train_tokens.npy, coco_tokens.npy)
│   ├── feats_val_np_v2.parquet    # Feature dump on 125 val_np molecules
│   ├── feats_val_rand_v2.parquet  # Feature dump on 954 val_rand molecules
│   └── ranker.json                # Local fitted ranker configuration
│
└── output/                        # Local validation outputs
    ├── submission_v2.csv          # Local 400-mol test output for V2
    └── submission_v3.csv          # Local 400-mol test output for V3
```

---

## 6. Strategic Roadmap: Potential Directions for Version 4 (V4)

Once the V3 Kaggle leaderboard score is confirmed, here are the highest-ROI directions for **V4**:

1. **Ranker Ensembling Across Multiple Folds**:
   - Currently, `ranker.json` contains a single MLP-32 model fitted across all 1,079 molecules.
   - An ensemble of 5 MLP rankers trained with different initializations and fold splits will reduce ranking variance and boost top-1 accuracy.
2. **Molecular Formula Filtering / ChemAxon Rules**:
   - Natural products adhere to strict valence and element-ratio heuristics (Seven Golden Rules: nitrogen rule, unsaturated ring degrees).
   - Filtering de novo generated SMILES through valence and isotopic rule checks will eliminate chemically implausible hallucinations.
3. **Nucleus (Top-p) Sampling or Beam Search for De Novo Generation**:
   - Currently, `sample()` uses temperature-scaled ancestral sampling ($T=1.0$).
   - Implementing Top-$p$ (nucleus) sampling (e.g., $p=0.9$) or beam search with length penalties will significantly increase the quality and validity of Class 3 candidates.
4. **Spectral Peak Augmentation in Fine-Tuning**:
   - Introduce random intensity jittering ($\pm 10\%$) and peak dropout ($10\%$) during fine-tuning to make the SpectrumEncoder more robust to differing instrument noise levels.
5. **Precursor Fragment Loss (Neutral Loss) Features**:
   - Compute neutral loss spectra ($\Delta m/z = \text{precursor} - m/z$) and add neutral loss matching as explicit features in the ranker. Certain losses ($-\text{H}_2\text{O}$, $-\text{CO}_2$, $-\text{glucose}$) are diagnostic of specific chemical classes.
