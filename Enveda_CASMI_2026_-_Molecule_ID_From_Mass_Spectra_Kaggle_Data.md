# Enveda CASMI 2026 — Molecule ID From Mass Spectra — Data

**Predict 2D chemical structures of small molecules detected in complex biological extracts**

Enveda · Featured Code Competition · 3 months to go

Source: <https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data>

---

## Dataset Description

Your task is to identify the chemical structure of an unknown molecule from its tandem mass spectrometry (MS/MS) spectra. For each molecule you submit up to 25 candidate structures as SMILES, ranked best-guess first. CASMI 2026 focuses on molecules that resemble those found in natural samples from plants, mammals, or microbes — confirmed natural products, hypothesised natural products, natural product analogs, and synthetic molecules that might plausibly occur in nature. A molecule may have been measured several times at different collision energies or as different adducts. **Predictions are made per molecule, not per spectrum**, so you must aggregate the evidence from all of a molecule's spectra into one ranked list.

---

## Files

- **train.parquet** — ~2.5m MS/MS spectra covering ~275k unique structures, with the structure given. One row per spectrum.
- **test.parquet** — the spectra to identify. One row per spectrum; the structure is not given. **Note:** the file you see is comprised of examples from the training dataset; it will be replaced by the hidden test set during re-run, and will be approximately the same size.
- **sample_submission.csv** — a valid submission in the correct format.

---

## Test set

~1,500 spectra of ~400 molecules, 1–16 spectra per molecule (median 3), all acquired on a Bruker timsTOF. Monoisotopic masses range from 157 to 1,159 Da (median 348).

Every scored molecule falls into one of three novelty classes, in order of increasing difficulty:

| Class | Definition |
| --- | --- |
| **1 — in public spectral libraries** | The structure has publicly available reference MS/MS spectra, so it may be found by similarity to the training spectra. |
| **2 — known structure, no public spectra** | No public spectra exist for this molecule, but the structure is in PubChem or the natural-products database COCONUT, so it is reachable by database retrieval. |
| **3 — novel structure** | The structure is not in PubChem at all and must be predicted de novo. |

To mimic a real complex mixture, the distribution over classes — and which molecule belongs to which class — is hidden for the duration of the competition.

The test spectra have had only minimal cleaning applied, because there is no standard way to curate mass spectra and we would rather not discard information on your behalf:

- Spectra whose measured precursor mass was inconsistent with the known structure and adduct were dropped.
- Spectra whose base peak was below 1,000 raw counts were dropped as likely noise.
- Peaks above precursor + 2 Da were removed — a fragment cannot outweigh its singly-charged precursor.

Test spectra use one of ten adducts: `[M+H]+`, `[M+NH4]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M+Na]+`, `[M+K]+`, `[M-H]-`, `[M-H2O-H]-`, `[M+CH2O2-H]-`, `[M+Cl]-`. The training set contains many more.

---

## Columns

Peak lists are stored as two aligned arrays: the i-th m/z pairs with the i-th intensity, with intensities normalised per spectrum so the largest peak equals 1.0. `train.parquet` and `test.parquet` use identical names for the shared columns, so preprocessing code ports directly.

### test.parquet (12 columns)

| Column | Description |
| --- | --- |
| `molecule_id` | Anonymous compound identifier. Predictions are made per `molecule_id`. |
| `spectrum_id` | Unique identifier for each spectrum. |
| `ms2_mzs` | Array of fragment m/z values. |
| `ms2_normalized_intensities` | Aligned array of fragment intensities; base peak = 1.0. |
| `base_peak_intensity` | Raw intensity of the largest peak before normalisation. More intense spectra usually contain less noise. |
| `adduct` | Precursor adduct, one of the seven above. |
| `ionization_mode` | `positive` or `negative`. |
| `instrument_type` | Always `timsTOF` — all test spectra share one platform. |
| `precursor_mz` | Measured precursor m/z. |
| `collision_energy_ev` | Collision energy in electron volts, as a list. A single value (`[20]`) is one acquisition; several (`[20, 40, 60, 80]`) means the spectrum is merged from acquisitions at different energies. |
| `collision_energy_orig` | The collision energy as originally recorded. |
| `collision_energy_orig_units` | Always `eV` in the test set. |

### train.parquet (18 columns)

The test columns above, plus:

| Column | Description |
| --- | --- |
| `normalized_smiles` | **The label.** The structure as an RDKit-standardised SMILES string. |
| `inchikey` / `inchikey14` | InChIKey of the structure and its first block, which identifies the 2D skeleton independent of stereochemistry. |
| `molecular_formula` | Formula of the neutral structure. |
| `ingest_lib` | Source library (see below). |
| `adduct_orig` | The adduct string as recorded by the source, before standardisation. |
| `precursor_error_ppm` | ppm error between the measured precursor m/z and the value implied by the labelled structure and adduct. Ships uncleaned, so it doubles as a label-quality signal. |
| `num_peaks` | Number of fragment peaks. |

Note that `instrument_type` in train is free text as recorded by each source (dozens of distinct strings, null where the source recorded none), and `base_peak_intensity` is null for libraries that shipped pre-normalised intensities. Only `enveda-180` and `enveda-np-examples` are uniformly labelled `timsTOF`, matching the test file.

---

## Training set

The training set is largely a convenience aggregation of publicly available data. You may also use any freely and publicly available external data — see the Rules for what qualifies.

Three things to understand before you use it:

- **Each library had different upstream preprocessing.** Different machines, different collision-energy units and conventions, free-text instrument descriptions, varying m/z precision.
- **Compounds overlap across libraries.** Exact duplicate spectra were removed — each spectrum appears once, credited to its primary source — but the same compound measured independently by several libraries remains.

| `ingest_lib` | Spectra | Unique structures | Notes |
| --- | --- | --- | --- |
| `enveda-180` | 1,153,785 | 182,941 | Enveda's published dataset, acquired on the same Bruker timsTOF as the test set. Consistent and instrument-matched, but the chemistry is synthetic drug-like screening compounds, a different region of chemical space from the test molecules. |
| `pluskal_ms2` | 527,581 | 46,821 | MSnLib from the Pluskal lab: Orbitrap spectra of commercial screening and bioactive-compound libraries, one consistent protocol at multiple collision energies. |
| `riken` | 347,171 | 15,892 | RIKEN's public libraries, strong focus on plant specialised metabolites. |
| `gnps` | 220,849 | 45,750 | The GNPS community libraries, the largest public collection of natural-product reference spectra — and the most heterogeneous, being community-contributed. |
| `massbank` | 101,727 | 9,180 | MassBank, a curated consortium library aggregating many labs and instrument types. |
| `mona` | 92,416 | 11,681 | MassBank of North America, community-hosted, maintained by the Fiehn Lab at UC Davis. |
| `spectraverse` | 50,933 | 9,631 | A recent harmonised aggregation of public libraries, including obscure ones absent from other aggregations. |
| `msdial` | 40,765 | 9,127 | Public collections distributed with the MS-DIAL software, including multi-instrument libraries. |
| `drug_plus` | 2,545 | 2,539 | A pharmaceutical LC-MS/MS database: ~2,500 drug substances, roughly one spectrum each, no collision-energy metadata. |
| `enveda-np-examples` | 1,151 | 250 | **The closest library to the test set.** 250 common natural products gathered on the same instruments and processed through the same pipeline as the test spectra, released to help you build a domain-adapted pipeline. The compounds are deliberately common and appear in other libraries too, so you can compare. |
| `masaryk` | 652 | 416 | A small library of chemical standards from RECETOX (Masaryk University). |

Structure counts sum to more than 275,810 because compounds recur across libraries.

---

## Collision energy

Collision energy governs how completely a molecule fragments: higher energy means more fragmentation and smaller fragments on average. Instruments report it differently — absolute electron volts (eV, beam-type instruments), normalised collision energy (NCE, a Thermo percentage scale), occasionally raw voltages, and often nothing at all. eV and NCE are different quantities on different scales, so the training set carries three columns:

- `collision_energy_orig` — the value exactly as the source recorded it, lossless, as a string: `40`, `[20 40 60]`, `35HCD`, `6V`. A multi-value entry means several single-energy acquisitions were merged into one spectrum.
- `collision_energy_orig_units` — `eV`, `NCE`, `V`, or `unknown`. Assigned from the declared string, then from documented library conventions (MSnLib is NCE, confirmed by its authors; Enveda-180 is eV), then from the instrument's analyser family.
- `collision_energy_ev` — our best-effort conversion to eV, matching the test file's convention. NCE is converted with the nominal Thermo formula (NCE × precursor_mz / 500 × charge factor), which is approximate because the true conversion depends on instrument tuning. Null where the unit is unresolved.

**If your model consumes collision energy, use `collision_energy_ev`** — it is the column designed to line up with the test set.

---

## Common curation methods

None of these is required and the best settings are yours to find, but they are common:

- **Relative-intensity floor.** Drop peaks below a threshold relative to the base peak; 2%, 1% and 0.1% are common. Very low peaks are often instrument noise.
- **Drop peaks above the precursor m/z.** A fragment cannot be heavier than the ion it came from, so peaks above the precursor (plus 1–2 Da for isotopes) are co-isolated contaminants or noise.
- **Minimum peak count.** Spectra with a handful of peaks carry little structural information; many pipelines drop those with fewer than ~5–6.
- **Maximum peak count.** Peak counts are heavy-tailed and some spectra have thousands. Keeping the top-N most intense (e.g. N=128) after the filters above is common. A classic library-search variant keeps the top ~6 peaks per 50 Da window, which preserves low-mass fragments a global top-N would discard.
- **Deisotoping.** Each real fragment can be accompanied by a carbon-13 companion ~1.0033 Da above it; these are sometimes removed so each fragment is represented once.
- **Intensity transforms.** Square-root or log scaling is common, since raw intensities span orders of magnitude and the largest peaks otherwise dominate any loss or similarity.

---

## Other resources

- **MIST-CF** and **SIRIUS** — learned and rule-based molecular-formula annotation from MS/MS. Predicting the molecular formula first and conditioning on it is a common method for narrowing down the search space of possible structures.
- **matchms** — an open library for working with mass spectra: cleaning, curation, and similarity.
- **GNPS propagated ("suspect") annotations** — structures transferred to unidentified spectra via spectral-network similarity. Excluded from `train.parquet` because the labels are inferred rather than measured.
- **MassIVE** — the community repository of raw metabolomics runs: billions of unannotated MS/MS spectra. No structure labels, but a natural corpus for self-supervised pretraining.
- **GeMS and DreaMS** — the DreaMS project released GeMS, a large curated corpus of unannotated spectra mined from MassIVE, plus pretrained spectrum-embedding models. A shortcut to repository-scale pretraining.
- **FragHub** — an open data repository that merges and harmonises the major open MS/MS libraries. Overlaps heavily with `train.parquet` but may contain some additional spectra either experimental or predicted.

---

## Dataset metadata

- **Files:** 3 (`sample_submission.csv`, `test.parquet`, `train.parquet`)
- **Size:** 3.04 GB
- **Type:** parquet, csv
- **License:** Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

### sample_submission.csv (43.62 kB)

2 columns — `molecule_id` (400 unique values) and `smiles` (1 unique value). Every row repeats the same placeholder prediction:

```
molecule_id,smiles
m_005e53,CCO;CCO;CCO;...;CCO   (25 repeats of CCO)
m_006153,CCO;CCO;CCO;...;CCO
m_00b5aa,CCO;CCO;CCO;...;CCO
...
```
