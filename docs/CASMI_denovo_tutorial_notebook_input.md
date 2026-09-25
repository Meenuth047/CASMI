# CASMI denovo tutorial notebook — Input

Author: **inversion** · 3 days ago · 1,992 views · 85 upvotes · 72 copies

Source: <https://www.kaggle.com/code/inversion/casmi-denovo-tutorial-notebook/input>

> **Note:** This page is Kaggle's notebook viewer, which loads the actual notebook cells (code + outputs) client-side via JavaScript. The saved HTML snapshot only captured the surrounding page — the input dataset description, file listing, and table of contents — not the notebook's code cells. To get the executed cells, download the `.ipynb` directly from Kaggle or use "Copy & Edit."

---

## Notebook Info

- **Competition:** Enveda CASMI 2026 - Molecule ID From Mass Spectra
- **Version:** 9 of 9
- **Runtime:** 1h 58m 31s · GPU T4 x2
- **Language:** Python
- **Tags:** GPU
- **Related notebook:** Dependency Installation Script

## Table of Contents (from the notebook's own headers)

- Data
  - load data
  - process data
  - datamodule class
- Model
- Train
- Predict

## Input Data

**Enveda CASMI 2026 - Molecule ID From Mass Spectra** — Predict 2D chemical structures of small molecules detected in complex biological extracts. Last updated 3 days ago.

### About this Competition

Your task is to identify the chemical structure of an unknown molecule from its tandem mass spectrometry (MS/MS) spectra. For each molecule you submit up to 25 candidate structures as SMILES, ranked best-guess first. CASMI 2026 focuses on molecules that resemble those found in natural samples from plants, mammals, or microbes — confirmed natural products, hypothesised natural products, natural product analogs, and synthetic molecules that might plausibly occur in nature. A molecule may have been measured several times at different collision energies or as different adducts. **Predictions are made per molecule, not per spectrum**, so you must aggregate the evidence from all of a molecule's spectra into one ranked list.

*(Full dataset description — files, columns, training-library breakdown, collision energy notes, curation methods, and other resources — matches the competition's Data page, already converted separately.)*

### Input Files

- **Input size:** 6.12 GB
- **Data Sources:** Enveda CASMI 2026 - Molecule ID From Mass Spectra
  - `sample_submission.csv`
  - `test.parquet`
  - `train.parquet`
- Additional input reference: `PM-134328238-at-09-14-2026-18-11-26`

## Competition Notebook

- **Competition:** Enveda CASMI 2026 - Molecule ID From Mass Spectra
- **Public Score:** 0.000
- **Best Score:** 0.000 (V9)
