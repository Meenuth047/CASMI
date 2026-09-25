# CASMI 2026 — Progress & Resume Guide

> **Competition**: [Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) · metric MRR@25 · deadline 14 Dec 2026
> **Last updated**: 21 Sep 2026, 17:45 — work paused for the day at the user's request. Nothing is running; GPU idle.

---

## 1. Where things stand

A complete V1 pipeline exists and produces a valid submission. It has **not been uploaded to Kaggle yet**.

| Item | State |
|---|---|
| Two decoder bugs from the tutorial (unshifted labels, last-token-only generation) | **Fixed** in the new model (`casmi/model.py`) and patched in `casmi_v1.py`, `casmi_v1_cpu_test.py` |
| MRR@25 scorer matching Kaggle (tautomer-canonical InChIKey block) | **Done** — `casmi/metric.py` |
| Adduct-aware mass + instrument calibration (+2.0 ppm pos / +0.6 ppm neg), ±10 ppm window | **Done** — `casmi/common.py` |
| Spectral library search (class 1) | **Done, validated** — `casmi/libsearch.py` (MRR 0.92–0.95 when the compound has public spectra) |
| COCONUT offline database (class 2) | **Done** — `work/db/coconut.parquet` (461,646 structures) + `casmi/candidates.py` |
| PubChem offline database (class 2) | **Not finished** — see §4 |
| Spectrum→SMILES + fingerprint model (class 2 ranking, class 3 de novo) | **V1 trained** (5 epochs, stopped for overfitting — see §3). Checkpoints `work/ckpt/model_ep{1,3,4,5}.pt` |
| Candidate ranker (small MLP, listwise loss, fitted on held-out molecules) | **Done (V1)** — `work/ranker.json` |
| Full pipeline on `test.parquet` | **Done** — `output/submission.csv` (400 rows, median 25 guesses, no fallbacks) |
| Kaggle notebook + assets folder | **Built** — `casmi_v1_kaggle.ipynb`, `kaggle_assets/` (283 MB) |
| Kaggle upload + submission | **Not done** — needs the user (see §5) |

### Validation results (held-out molecules the model never trained on)

125 natural products measured on the same timsTOF pipeline as the hidden test (`val_np`), each scored under a simulated novelty class. 5-fold cross-validated ranker, epoch-3 model, COCONUT only (no PubChem):

| Simulated class | MRR@25 |
|---|---|
| c1 — public spectra exist (library keeps the compound from *other* labs) | **~0.90** (library score alone: 0.94) |
| c2 — no spectra, structure in COCONUT | **~0.53** (was 0.19 with hand weights; 0.56 with the epoch-2 model) |
| c3 — novel, only de novo generation can find it | **~0.18** (0.24 with the epoch-2 model) |

The old tutorial code scores 0.000. The real leaderboard score depends on the hidden class mix.

`test.parquet` locally is a placeholder made of 400 `enveda-180` **training** molecules, so the local score on it (MRR 0.995) only proves the pipeline runs — it says nothing about the real test.

---

## 2. Code map

```
casmi/
  common.py        adduct parser + masses, calibrated neutral mass, metric key, Morgan FP, peak prep, SMILES tokenizer
  metric.py        MRR@25 replica + Kaggle format checks      (python -m casmi.metric sub.csv truth.csv)
  prep.py          train.parquet -> work/structures.parquet + work/spectra/*.npz (2.04M spectra, 275k structures)
  make_val.py      held-out query sets: work/val_np.parquet (125 mol), work/val_rand.parquet (954 mol) + truth csv
  model.py         encoder (peaks + adduct/CE/mass token) -> fingerprint head + SMILES decoder (KV-cache generation)
  train.py         training loop (bf16, cosine LR, spectrum augmentation, per-epoch validation)
  infer.py         ModelScorer: fingerprints, candidate log-likelihood, de novo sampling
  libsearch.py     spectral library search over train.parquet (numpy only)          [built by subagent, validated]
  candidates.py    CandidateDB: fast mass-window queries on the parquet databases   [built by subagent]
  pipeline.py      merges library + COCONUT/PubChem + de novo candidates, features, ranker, top-25 dedupe
  validate.py      runs the pipeline under simulated classes c1/c2/c3 and reports MRR@25
  dump_features.py dumps candidate feature tables -> work/feats_*.parquet (for offline ranker work)
  ranker_lab.py    cross-validated ranker comparison; fits work/ranker.json
  run_test.py      test.parquet -> submission.csv (same entry point locally and on Kaggle)
  prep_v2.py       V2 data: randomized SMILES + COCONUT structure-only set -> work/v2/   (already run)
scripts/
  train_when_gpu_free.sh     waits for train_yolo11s.py to exit + GPU memory free, then trains
  snapshot_epochs.sh         keeps a copy of every epoch's weights
  package_assets.py          fills kaggle_assets/ and rebuilds the notebook
  build_kaggle_notebook.py   generates casmi_v1_kaggle.ipynb (code embedded via %%writefile)
  build_pubchem_hf.py        PubChem builder from the mirrored dump (see §4)
```

Environments: `~/casmi-gpu-venv/bin/python` (GPU; thin venv over the `training` conda env's torch 2.10 + RDKit 2026.3.3) and `~/miniconda3/envs/casmi/bin/python` (CPU; RDKit, pandas, pyarrow, sklearn). Run modules from the project root: `python -m casmi.<module>`.

---

## 3. Key finding from V1 training — the model memorises

Training loss kept falling (0.29 → 0.14) but loss on **unseen structures** bottomed at epoch 2 and then rose
(val_np CE 0.309 / **0.289** / 0.303 / 0.292 / 0.318; fingerprint cosine 0.62 → 0.57). The run was stopped after epoch 5 of 12.
Memorisation is harmless for class 1 (those molecules are in the training set) but hurts classes 2 and 3.
Epoch-2 weights were not kept (snapshotting started at epoch 3); saved: epochs 1, 3, 4, 5.

**V2 training plan (data already prepared in `work/v2/`, trainer changes NOT yet written):**
- dropout 0.1 → 0.2; cap spectra per structure per epoch (balanced sampling)
- randomized SMILES targets (4 variants per structure; 99.4% differ from canonical) so string memorisation is useless
- ~25% of each batch = structure-only examples from COCONUT (459,480 structures, validation structures removed):
  no peaks, only the mass/adduct token → teaches the decoder natural-product chemistry and mass conditioning
- new vocab (74 tokens, `work/v2/vocab.json`) — store the vocab inside the checkpoint and load it in `ModelScorer`
- save the best checkpoint by validation loss, not the last one
- estimated ~2 h on the RTX 4090

---

## 4. PubChem status

NCBI throttled this machine to <100 KB/s, so the three official dumps were unreachable. The subagent found a mirrored
full-PubChem SMILES snapshot and downloaded it: **`work/db/tmp/pubchem_hf.tar.xz` (698 MB) — keep this file.**
`scripts/build_pubchem_hf.py` reads it from stdin, computes mass / InChIKey block with RDKit, filters
(100–1300 Da, C,H,N,O,P,S,halogens only, neutral, single component, no isotopes), dedupes and writes
`work/db/pubchem.parquet`. It was ~42% through (50M of ~119M rows, ~28k rows/s, all in RAM) when stopped — it must be
re-run from the start (~70 min, CPU only, no download). The exact launch command the subagent used was not recorded;
read the script header and pipe the decompressed archive member into it. Then verify: row count, quercetin / caffeine /
reserpine found, coverage of the 250 np-examples structures > 90%, `query_many` timing.
The pipeline picks up `pubchem.parquet` automatically when it exists (and the ranker must then be re-fitted, because
PubChem adds thousands of distractor candidates).

---

## 5. How to submit V1 to Kaggle (user action)

1. Upload the assets folder as a private dataset (283 MB):
   `kaggle datasets create -p kaggle_assets` (metadata already points to `Deepak-airbotix/casmi-v1-assets`)
2. On the competition page create a notebook from `casmi_v1_kaggle.ipynb`; attach the competition data **and** that dataset; GPU T4; internet off.
   RDKit installs offline from the wheel inside the dataset if the environment lacks it.
3. Run all (expected well under 1 hour) → Submit `submission.csv`.

Current `kaggle_assets/model.pt` = epoch-4 weights; `ranker.json` was fitted on epoch-3 features (small mismatch, acceptable for a first submission).

---

## 6. Resume checklist for the next session

1. **Pick the best V1 checkpoint**: `python -m casmi.dump_features --no-pubchem --tag _ep1 --model work/ckpt/model_ep1.pt` (and `_ep4`), compare with `casmi.ranker_lab`-style CV. Epoch 1 may generalise better than 4/5.
2. **Final ranker**: 5-member MLP-32 ensemble, l2 = 1e-2 was the most stable in CV (np c1 0.91 / c2 0.52 / c3 0.19). `pipeline.rank_score` needs a small extension to average ensemble members. Analog-similarity features were neutral.
3. Re-run `scripts/package_assets.py --model <best>` and `python -m casmi.run_test`.
4. **PubChem**: re-run the build (§4), re-dump features with PubChem, re-fit the ranker.
5. **V2 model**: implement the trainer changes in §3 and train (check the GPU is free first).
6. Upload + submit (§5) — ideally submit V1 early to confirm the notebook runs on Kaggle.

Open risks: hidden test has far more candidates per mass window than validation (median 80 vs 9–13); the true tautomer
canonicalisation settings of the host metric are assumed to be RDKit defaults; disk is tight (21 GB free, something
outside this project consumed ~20 GB during the session).
