"""
Spectral library search for the CASMI 2026 pipeline (class-1 molecules).

For a molecule we only have its MS/MS spectra.  ``estimate_molecule_mass`` turns the
precursor m/z values into a calibrated neutral mass; ``SpectralLibrary`` holds every
training spectrum whose *structure* mass falls into one of the query mass windows, and
``SpectralLibrary.search`` ranks the structures in one window by how well their library
spectra match the query spectra.

numpy + pandas + pyarrow only -- no numba, no rdkit at search time, so the same file runs
unchanged in a Kaggle notebook.

    from casmi.libsearch import SpectralLibrary, SearchConfig, estimate_molecule_mass

    mass = estimate_molecule_mass(prec_mzs, adducts, modes)
    lo, hi = mass * (1 - 10e-6), mass * (1 + 10e-6)
    lib = SpectralLibrary.from_train('train.parquet', structures_df, [(lo, hi)])
    hits = lib.search(query_spectra, lo, hi)
"""

import os
import time
import numpy as np
import pandas as pd

try:                                    # pyarrow is only needed to build the library
    import pyarrow.parquet as pq
except Exception:                       # pragma: no cover
    pq = None


# ─────────────────────────────────────────────────────────────────────
# Adduct arithmetic (duplicated from casmi.common so this module has no rdkit import)
# ─────────────────────────────────────────────────────────────────────

ELECTRON_MASS = 0.00054857990907

# (n_mol, mass_shift, charge); m/z = (n_mol * M + shift) / |charge|
_ADDUCT_TABLE = {
    '[M+H]+':        (1,   1.00727645, 1),
    '[M+NH4]+':      (1,  18.03382555, 1),
    '[M-H2O+H]+':    (1, -17.00328247, 1),
    '[M-2H2O+H]+':   (1, -35.01384139, 1),
    '[M+Na]+':       (1,  22.98922123, 1),
    '[M+K]+':        (1,  38.96315773, 1),
    '[M-H]-':        (1,  -1.00727645, -1),
    '[M-H2O-H]-':    (1, -19.01784037, -1),
    '[M+CH2O2-H]-':  (1,  44.99820285, -1),
    '[M+Cl]-':       (1,  34.96940140, -1),
}

# Everything else that occurs in the training libraries is parsed from the string.
_PT_MASS = {
    'H': 1.00782503207, 'D': 2.0141017778, 'C': 12.0, 'N': 14.0030740048, 'O': 15.9949146196,
    'F': 18.99840322, 'Na': 22.9897692809, 'Mg': 23.985041700, 'Si': 27.9769265325,
    'P': 30.97376163, 'S': 31.97207100, 'Cl': 34.96885268, 'K': 38.96370668,
    'Ca': 39.96259098, 'Fe': 55.9349375, 'Ni': 57.9353429, 'Cu': 62.9295975,
    'Zn': 63.9291422, 'Br': 78.9183371, 'I': 126.904473, 'Li': 7.01600455,
    'Ac': 42.01056468,          # acetyl/acetic-acid shorthand used by some libraries
    'FA': 46.00547931, 'Hac': 60.02112937, 'TFA': 113.99286,
}

_adduct_cache = dict(_ADDUCT_TABLE)


def _formula_mass(formula):
    import re
    total, consumed = 0.0, 0
    for sym, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
        if not sym:
            continue
        if sym not in _PT_MASS:
            return None
        total += _PT_MASS[sym] * (int(count) if count else 1)
        consumed += len(sym) + len(count)
    return total if consumed == len(formula) else None


def parse_adduct(adduct):
    """'[2M+Na-2H]-' -> (n_mol, mass_shift, charge); None when unparsable."""
    if adduct in _adduct_cache:
        return _adduct_cache[adduct]
    import re
    result = None
    m = re.match(r'^\[(\d*)M([^\]]*)\](\d*)([+-])$', adduct.replace(' ', '')) if isinstance(adduct, str) else None
    if m:
        n_mol = int(m.group(1)) if m.group(1) else 1
        charge = (int(m.group(3)) if m.group(3) else 1) * (1 if m.group(4) == '+' else -1)
        body, shift, consumed, ok = m.group(2), 0.0, 0, True
        for sign, mult, formula in re.findall(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)', body):
            fm = _formula_mass(formula)
            if fm is None:
                ok = False
                break
            shift += (1 if sign == '+' else -1) * (int(mult) if mult else 1) * fm
            consumed += len(sign) + len(mult) + len(formula)
        if ok and consumed == len(body):
            result = (n_mol, shift - charge * ELECTRON_MASS, charge)
    _adduct_cache[adduct] = result
    return result


def neutral_mass_from_mz(precursor_mz, adduct):
    parsed = parse_adduct(adduct)
    if parsed is None:
        return float('nan')
    n_mol, shift, charge = parsed
    return (precursor_mz * abs(charge) - shift) / n_mol


# ─────────────────────────────────────────────────────────────────────
# Mass calibration
# ─────────────────────────────────────────────────────────────────────

# Measured-minus-theoretical neutral mass on the Bruker timsTOF used for the hidden test
# set.  On enveda-np-examples alone (same instrument and pipeline as the test set) the
# median is +1.89 ppm positive / +0.45 ppm negative; casmi.common uses +2.0 / +0.6 measured
# over both timsTOF libraries.  These values are only a fallback -- see estimate_molecule_mass.
PPM_BIAS = {'positive': 2.0, 'negative': 0.6}
DEFAULT_MASS_PPM = 10.0


def _mode_key(mode):
    if mode is None:
        return 'positive'
    m = str(mode).strip().lower()
    if m.startswith('n') or m in ('-', 'neg'):
        return 'negative'
    return 'positive'


def estimate_molecule_mass(precursor_mzs, adducts, ion_modes=None):
    """Calibrated neutral mass of one molecule from all of its spectra (median, robust).

    Delegates to ``casmi.common.molecule_neutral_mass`` (the shared, jointly-calibrated
    implementation) when it is importable; the fallback below repeats the same arithmetic
    so this module also works on its own.  NaN when no spectrum has a parsable adduct.
    """
    try:
        from casmi.common import molecule_neutral_mass
        return molecule_neutral_mass(list(precursor_mzs), list(adducts), calibrate=True)
    except Exception:
        pass
    modes = ion_modes if ion_modes is not None else [None] * len(list(precursor_mzs))
    out = []
    for mz, ad, mode in zip(precursor_mzs, adducts, modes):
        m = neutral_mass_from_mz(float(mz), ad)
        if not (np.isfinite(m) and m > 0):
            continue
        if mode is None:
            parsed = parse_adduct(ad)
            mode = 'positive' if (parsed and parsed[2] > 0) else 'negative'
        out.append(m * (1.0 - PPM_BIAS[_mode_key(mode)] * 1e-6))
    return float(np.median(out)) if out else float('nan')


def mass_window(mass, ppm=DEFAULT_MASS_PPM):
    d = mass * ppm * 1e-6
    return mass - d, mass + d


# ─────────────────────────────────────────────────────────────────────
# Search configuration
# ─────────────────────────────────────────────────────────────────────

class SearchConfig(object):
    """Everything that can be tuned about the similarity.  Cheap to change (see `set_config`).

    The defaults are the configuration chosen by the validation sweep in casmi.libsearch_eval
    (see work/libsearch/REPORT.md): sqrt-intensity cosine with greedy one-to-one matching at
    0.02 Da, precursor region removed, a 0.5% relative-intensity floor, the 60 most intense
    peaks, and per-molecule aggregation by the mean of the two best query spectra.
    """

    def __init__(self,
                 tol_da=0.02,               # peak-matching tolerance in Da
                 tol_ppm=0.0,               # extra m/z-proportional tolerance (added to tol_da)
                 power=0.5,                 # intensity transform i**power (1.0 = none, 0.5 = sqrt)
                 mz_power=0.0,              # optional m/z weighting: i**power * mz**mz_power
                 min_rel=0.005,             # relative-intensity floor applied to both sides
                 max_peaks=60,              # top-N peaks kept per spectrum
                 drop_precursor=1.5,        # remove peaks within this many Da of the precursor (0 = keep)
                 modified=False,            # also match neutral-loss-shifted peaks (modified cosine)
                 min_matches=2,             # cosine forced to 0 below this many matched peaks
                 same_mode_only=True,       # only compare spectra of the same ionization mode
                 same_adduct_bonus=0.0,     # multiply cross-adduct cosines by (1 - this)
                 agg='mean_top2',           # 'max' | 'mean_qmax' | 'mean_qmax_all' | 'mean_topK'
                 ):
        self.tol_da = float(tol_da)
        self.tol_ppm = float(tol_ppm)
        self.power = float(power)
        self.mz_power = float(mz_power)
        self.min_rel = float(min_rel)
        self.max_peaks = int(max_peaks)
        self.drop_precursor = float(drop_precursor)
        self.modified = bool(modified)
        self.min_matches = int(min_matches)
        self.same_mode_only = bool(same_mode_only)
        self.same_adduct_bonus = float(same_adduct_bonus)
        self.agg = str(agg)

    def copy(self, **kw):
        c = SearchConfig(**self.__dict__)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def __repr__(self):
        d = self.__dict__
        return 'SearchConfig(' + ', '.join('%s=%r' % (k, d[k]) for k in sorted(d)) + ')'


DEFAULT_CONFIG = SearchConfig()


# ─────────────────────────────────────────────────────────────────────
# Peak preprocessing
# ─────────────────────────────────────────────────────────────────────

def prep_spectrum(mzs, intensities, precursor_mz, max_peaks=256, min_rel=0.0, drop_precursor=0.0):
    """Clean one spectrum -> (mz, intensity) float32 arrays sorted by m/z, base peak = 1."""
    mz = np.asarray(mzs, dtype=np.float64).ravel()
    it = np.asarray(intensities, dtype=np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    keep = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0)
    if precursor_mz and np.isfinite(precursor_mz):
        keep &= mz <= precursor_mz + 2.0
        if drop_precursor > 0:
            keep &= np.abs(mz - precursor_mz) > drop_precursor
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    it = it / it.max()
    if min_rel > 0:
        keep = it >= min_rel
        mz, it = mz[keep], it[keep]
        if len(mz) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
    if len(mz) > max_peaks:
        top = np.argpartition(it, -max_peaks)[-max_peaks:]
        mz, it = mz[top], it[top]
    order = np.argsort(mz, kind='stable')
    return mz[order].astype(np.float32), it[order].astype(np.float32)


def _transform(mz, it, cfg):
    """Apply the config's intensity transform; returns float64 weights."""
    w = np.asarray(it, dtype=np.float64)
    if cfg.power != 1.0:
        w = w ** cfg.power
    if cfg.mz_power != 0.0:
        w = w * (np.asarray(mz, dtype=np.float64) ** cfg.mz_power)
    return w


# ─────────────────────────────────────────────────────────────────────
# Core similarity: one query spectrum vs. a block of library spectra
# ─────────────────────────────────────────────────────────────────────

def _ranges(starts, counts):
    """Concatenated index ranges [starts[i], starts[i]+counts[i]) without a python loop."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    counts = counts.astype(np.int64)
    out = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return out + np.repeat(starts.astype(np.int64), counts)


def _segment_max(values, starts, counts):
    """Per-segment maximum of `values` along the last axis; empty segments give 0."""
    out = np.zeros(values.shape[:-1] + (len(starts),), dtype=values.dtype)
    ok = counts > 0
    if ok.any():
        red = np.maximum.reduceat(values, starts[ok], axis=-1)
        out[..., ok] = red
    return out


def _segment_any(values, starts, counts):
    out = np.zeros(values.shape[:-1] + (len(starts),), dtype=bool)
    ok = counts > 0
    if ok.any():
        red = np.logical_or.reduceat(values, starts[ok], axis=-1)
        out[..., ok] = red
    return out


def _match_pairs(qmz, gmz, tol_da, tol_ppm):
    """Candidate (query peak, library peak) index pairs with |dmz| <= tol.

    `gmz` must be sorted ascending.  Fully vectorised: one searchsorted per side.
    """
    tol = tol_da + qmz * tol_ppm * 1e-6
    lo = np.searchsorted(gmz, qmz - tol, side='left')
    hi = np.searchsorted(gmz, qmz + tol, side='right')
    cnt = hi - lo
    total = int(cnt.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    qi = np.repeat(np.arange(len(qmz), dtype=np.int64), cnt)
    # ranges [lo_i, hi_i) concatenated without a python loop
    starts = np.repeat(lo.astype(np.int64), cnt)
    offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return qi, starts + offs


def _greedy_one_to_one(qi, gj, prod, spec_of_peak, n_qpeaks):
    """Approximate greedy one-to-one matching, vectorised.

    Pairs are visited in descending intensity product; each (library spectrum, query peak)
    slot and each library peak may be used at most once.  This reproduces exact greedy
    matching whenever the winning pair of a peak is unambiguous, which is the normal case
    at the tolerances used here.
    """
    if len(qi) == 0:
        return qi, gj, prod
    order = np.argsort(-prod, kind='stable')
    qi, gj, prod = qi[order], gj[order], prod[order]
    # each query peak may be used once per library spectrum
    key = spec_of_peak[gj] * np.int64(n_qpeaks) + qi
    _, first = np.unique(key, return_index=True)
    first.sort()
    qi, gj, prod = qi[first], gj[first], prod[first]
    # each library peak may be used once (peak ids are globally unique)
    _, first = np.unique(gj, return_index=True)
    first.sort()
    return qi[first], gj[first], prod[first]


class _PeakBlock(object):
    """All peaks of a set of library spectra, concatenated and globally sorted by m/z."""

    def __init__(self, mz, weight, spec_idx):
        order = np.argsort(mz, kind='stable')
        self.mz = np.ascontiguousarray(mz[order], dtype=np.float64)
        self.weight = np.ascontiguousarray(weight[order], dtype=np.float64)
        self.spec = np.ascontiguousarray(spec_idx[order], dtype=np.int64)


def _cosine_block(qmz, qw, block, n_spec, cfg, shifts=None):
    """Cosine numerators + matched-peak counts of one query spectrum against `n_spec` library spectra.

    `shifts` (optional, length n_spec) gives a per-library-spectrum precursor difference for
    modified-cosine matching.  Returns (numerator[n_spec], n_matched[n_spec]).
    """
    num = np.zeros(n_spec, dtype=np.float64)
    cnt = np.zeros(n_spec, dtype=np.float64)
    if len(qmz) == 0 or len(block.mz) == 0:
        return num, cnt

    qi, gj = _match_pairs(qmz, block.mz, cfg.tol_da, cfg.tol_ppm)

    if shifts is not None:
        # neutral-loss channel: query peak at qmz matches a library peak at qmz - shift
        uniq = np.unique(shifts)
        uniq = uniq[np.abs(uniq) > cfg.tol_da]
        for sh in uniq:
            qi2, gj2 = _match_pairs(qmz - sh, block.mz, cfg.tol_da, cfg.tol_ppm)
            if len(qi2) == 0:
                continue
            ok = shifts[block.spec[gj2]] == sh
            if ok.any():
                qi = np.concatenate([qi, qi2[ok]])
                gj = np.concatenate([gj, gj2[ok]])
        if len(qi):
            key = block.spec[gj] * np.int64(len(qmz) + 1) + qi
            dedup = np.unique(key, return_index=True)[1]
            qi, gj = qi[dedup], gj[dedup]

    if len(qi) == 0:
        return num, cnt
    prod = qw[qi] * block.weight[gj]
    qi, gj, prod = _greedy_one_to_one(qi, gj, prod, block.spec, len(qmz))
    s = block.spec[gj]
    num += np.bincount(s, weights=prod, minlength=n_spec)[:n_spec]
    cnt += np.bincount(s, minlength=n_spec)[:n_spec]
    return num, cnt


# ─────────────────────────────────────────────────────────────────────
# The library
# ─────────────────────────────────────────────────────────────────────

_CHEAP_COLS = ['ingest_lib', 'inchikey14', 'adduct', 'ionization_mode',
               'precursor_mz', 'precursor_error_ppm']
_PEAK_COLS = ['ms2_mzs', 'ms2_normalized_intensities']

RAW_MAX_PEAKS = 256          # how many peaks are stored per library spectrum
RAW_MIN_REL = 0.002          # storage-time intensity floor (well below any config's floor)


class SpectralLibrary(object):
    """Training spectra restricted to a set of neutral-mass windows, ready for searching."""

    def __init__(self, ik14, smiles, struct_mass, n_struct_spectra,
                 spec_struct, spec_prec, spec_mode, spec_adduct, spec_lib,
                 peak_mz, peak_int, peak_off, config=None, build_seconds=0.0):
        self.ik14 = np.asarray(ik14, dtype=object)              # [n_struct]
        self.smiles = np.asarray(smiles, dtype=object)
        self.struct_mass = np.asarray(struct_mass, dtype=np.float64)
        self.n_struct_spectra = np.asarray(n_struct_spectra, dtype=np.int64)
        self.spec_struct = np.asarray(spec_struct, dtype=np.int32)   # [n_spec] -> struct row
        self.spec_prec = np.asarray(spec_prec, dtype=np.float64)
        self.spec_mode = np.asarray(spec_mode, dtype=np.int8)        # +1 positive, -1 negative
        self.spec_adduct = np.asarray(spec_adduct, dtype=object)
        self.spec_lib = np.asarray(spec_lib, dtype=object)
        self.peak_mz = np.asarray(peak_mz, dtype=np.float32)
        self.peak_int = np.asarray(peak_int, dtype=np.float32)
        self.peak_off = np.asarray(peak_off, dtype=np.int64)         # [n_spec + 1]
        self.build_seconds = float(build_seconds)
        self._mass_order = np.argsort(self.struct_mass, kind='stable')
        self._sorted_mass = self.struct_mass[self._mass_order]
        self._spec_order = np.argsort(self.spec_struct, kind='stable')
        ss = self.spec_struct[self._spec_order]
        self._spec_start = np.searchsorted(ss, np.arange(len(self.ik14) + 1))
        self.config = None
        self.set_config(config or DEFAULT_CONFIG)

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def from_train(cls, train_path, structures_df, mass_windows, exclude_libs=(),
                   max_ppm_error=20.0, include_libs=None, config=None,
                   max_peaks=RAW_MAX_PEAKS, min_rel=RAW_MIN_REL, verbose=True,
                   row_groups=None):
        """Stream train.parquet and keep the spectra of structures inside `mass_windows`.

        structures_df: DataFrame with at least ik14, smiles, mass (and optionally n_spectra).
        mass_windows : list of (lo, hi) neutral-mass intervals; overlapping ones are merged.
        exclude_libs : ingest_lib values to drop (e.g. the validation library itself).
        max_ppm_error: rows with |precursor_error_ppm| above this (or NaN) are dropped --
                       they have unreliable adduct/structure labels.
        """
        if pq is None:
            raise RuntimeError('pyarrow is required to build a SpectralLibrary')
        t0 = time.time()
        exclude_libs = set(exclude_libs or ())
        include_libs = set(include_libs) if include_libs is not None else None

        lo, hi = _merge_windows(mass_windows)
        s_mass = structures_df['mass'].to_numpy(dtype=np.float64)
        inside = _in_windows(s_mass, lo, hi)
        sel = structures_df.loc[inside].reset_index(drop=True)
        if verbose:
            print('[libsearch] %d structures inside %d mass windows' % (len(sel), len(lo)), flush=True)

        ik_index = {k: i for i, k in enumerate(sel['ik14'].tolist())}
        n_struct_spectra = (sel['n_spectra'].to_numpy(dtype=np.int64)
                            if 'n_spectra' in sel.columns else np.zeros(len(sel), np.int64))

        spec_struct, spec_prec, spec_mode, spec_adduct, spec_lib = [], [], [], [], []
        mz_parts, int_parts, lens = [], [], []

        pf = pq.ParquetFile(train_path)
        rgs = range(pf.metadata.num_row_groups) if row_groups is None else list(row_groups)
        for rg in rgs:
            meta = pf.read_row_group(rg, columns=_CHEAP_COLS).to_pandas()
            keep = np.asarray(meta['inchikey14'].isin(ik_index).to_numpy(), dtype=bool).copy()
            if exclude_libs:
                keep &= ~meta['ingest_lib'].isin(exclude_libs).to_numpy()
            if include_libs is not None:
                keep &= meta['ingest_lib'].isin(include_libs).to_numpy()
            err = meta['precursor_error_ppm'].to_numpy(dtype=np.float64)
            keep &= np.isfinite(err) & (np.abs(err) <= max_ppm_error)
            idx = np.flatnonzero(keep)
            if verbose:
                print('[libsearch] row group %2d: %6d / %d kept' % (rg, len(idx), len(meta)), flush=True)
            if len(idx) == 0:
                continue
            meta = meta.iloc[idx]
            tab = pf.read_row_group(rg, columns=_PEAK_COLS).take(idx)
            mzs_list = tab.column(0).combine_chunks()
            int_list = tab.column(1).combine_chunks()
            m_off = np.asarray(mzs_list.offsets)
            m_val = np.asarray(mzs_list.values)
            i_val = np.asarray(int_list.values)
            i_off = np.asarray(int_list.offsets)
            del tab

            precs = meta['precursor_mz'].to_numpy(dtype=np.float64)
            struct_ids = [ik_index[k] for k in meta['inchikey14']]
            modes = [1 if _mode_key(m) == 'positive' else -1 for m in meta['ionization_mode']]
            adducts = meta['adduct'].tolist()
            libs = meta['ingest_lib'].tolist()

            for k in range(len(precs)):
                mz, it = prep_spectrum(m_val[m_off[k]:m_off[k + 1]], i_val[i_off[k]:i_off[k + 1]],
                                       precs[k], max_peaks=max_peaks, min_rel=min_rel)
                if len(mz) < 2:
                    continue
                mz_parts.append(mz)
                int_parts.append(it)
                lens.append(len(mz))
                spec_struct.append(struct_ids[k])
                spec_prec.append(precs[k])
                spec_mode.append(modes[k])
                spec_adduct.append(adducts[k])
                spec_lib.append(libs[k])
            del m_val, i_val, meta

        if lens:
            peak_mz = np.concatenate(mz_parts)
            peak_int = np.concatenate(int_parts)
            peak_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        else:
            peak_mz = np.zeros(0, np.float32)
            peak_int = np.zeros(0, np.float32)
            peak_off = np.zeros(1, np.int64)
        dt = time.time() - t0
        if verbose:
            print('[libsearch] kept %d spectra / %d peaks in %.1f s' % (len(lens), len(peak_mz), dt), flush=True)
        return cls(sel['ik14'].to_numpy(), sel['smiles'].to_numpy(), s_mass[inside], n_struct_spectra,
                   spec_struct, spec_prec, spec_mode, spec_adduct, spec_lib,
                   peak_mz, peak_int, peak_off, config=config, build_seconds=dt)

    # ── persistence (validation convenience; not needed on Kaggle) ──

    def save(self, path):
        np.savez(path, ik14=self.ik14, smiles=self.smiles, struct_mass=self.struct_mass,
                 n_struct_spectra=self.n_struct_spectra, spec_struct=self.spec_struct,
                 spec_prec=self.spec_prec, spec_mode=self.spec_mode, spec_adduct=self.spec_adduct,
                 spec_lib=self.spec_lib, peak_mz=self.peak_mz, peak_int=self.peak_int,
                 peak_off=self.peak_off, build_seconds=self.build_seconds)
        return path

    @classmethod
    def load(cls, path, config=None):
        d = np.load(path, allow_pickle=True)
        return cls(d['ik14'], d['smiles'], d['struct_mass'], d['n_struct_spectra'],
                   d['spec_struct'], d['spec_prec'], d['spec_mode'], d['spec_adduct'],
                   d['spec_lib'], d['peak_mz'], d['peak_int'], d['peak_off'],
                   config=config, build_seconds=float(d['build_seconds']))

    def drop_libs(self, libs):
        """A copy of this library without the given ingest_lib values (validation helper)."""
        keep = ~np.isin(self.spec_lib, list(libs))
        idx = np.flatnonzero(keep)
        counts = self.peak_off[idx + 1] - self.peak_off[idx]
        gather = _ranges(self.peak_off[idx], counts)
        off = np.zeros(len(idx) + 1, np.int64)
        np.cumsum(counts, out=off[1:])
        return SpectralLibrary(self.ik14, self.smiles, self.struct_mass, self.n_struct_spectra,
                               self.spec_struct[idx], self.spec_prec[idx], self.spec_mode[idx],
                               self.spec_adduct[idx], self.spec_lib[idx],
                               self.peak_mz[gather], self.peak_int[gather], off,
                               config=self.config, build_seconds=self.build_seconds)

    # ── config ──────────────────────────────────────────────────────

    def set_config(self, config):
        """(Re)apply the intensity transform / peak filters.  Fully vectorised, no file access."""
        self.config = config
        cfg = config
        n = len(self.spec_prec)
        off = self.peak_off
        n_peaks = len(self.peak_mz)
        if n == 0 or n_peaks == 0:
            self._cmz = np.zeros(0)
            self._cw = np.zeros(0)
            self._coff = np.zeros(n + 1, np.int64)
            self._cnorm = np.zeros(n)
            return self

        counts0 = off[1:] - off[:-1]
        owner = np.repeat(np.arange(n, dtype=np.int64), counts0)
        mz = self.peak_mz.astype(np.float64)
        it = self.peak_int.astype(np.float64)

        keep = np.ones(n_peaks, bool)
        if cfg.drop_precursor > 0:
            keep &= np.abs(mz - self.spec_prec[owner]) > cfg.drop_precursor
        if cfg.min_rel > 0:
            # renormalise to the base peak of what survived, then apply the relative floor
            masked = np.where(keep, it, 0.0)
            mx = _segment_max(masked, off[:-1], counts0)
            keep &= it >= cfg.min_rel * np.maximum(mx[owner], 1e-30)
        if cfg.max_peaks > 0:
            cnt = np.bincount(owner[keep], minlength=n) if keep.any() else np.zeros(n, np.int64)
            if (cnt > cfg.max_peaks).any():
                idx = np.flatnonzero(keep)
                order = np.lexsort((-it[idx], owner[idx]))     # by spectrum, intensity desc
                s_owner = owner[idx][order]
                starts = np.searchsorted(s_owner, np.arange(n))
                rank = np.arange(len(order), dtype=np.int64) - starts[s_owner]
                drop = idx[order[rank >= cfg.max_peaks]]
                keep[drop] = False

        mz, it, owner = mz[keep], it[keep], owner[keep]
        lens = np.bincount(owner, minlength=n)
        coff = np.zeros(n + 1, np.int64)
        np.cumsum(lens, out=coff[1:])
        # peaks are already in ascending-m/z order inside each spectrum and `keep` preserves it
        self._cmz = mz
        self._cw = _transform(mz, it, cfg)
        self._coff = coff
        sq = self._cw * self._cw
        self._cnorm = np.sqrt(np.bincount(owner, weights=sq, minlength=n))
        return self

    # ── search ──────────────────────────────────────────────────────

    def candidates(self, mass_lo, mass_hi):
        """Row indices (into self.ik14) of the structures inside the mass window."""
        a = np.searchsorted(self._sorted_mass, mass_lo, side='left')
        b = np.searchsorted(self._sorted_mass, mass_hi, side='right')
        return np.sort(self._mass_order[a:b])

    def _spectra_of(self, struct_rows):
        starts = self._spec_start[struct_rows]
        counts = self._spec_start[struct_rows + 1] - starts
        return self._spec_order[_ranges(starts, counts)]

    def search(self, query_spectra, mass_lo, mass_hi, exclude_libs=(), exclude_ik14=()):
        """Rank the structures in [mass_lo, mass_hi] by spectral similarity to `query_spectra`.

        query_spectra: list of dicts with keys mzs, intensities, precursor_mz, adduct,
                       ionization_mode -- ALL spectra of one molecule.
        exclude_libs : ingest_lib names whose spectra are ignored for THIS query only
                       (e.g. the library the validation query itself came from).
        exclude_ik14 : structures whose spectra are ignored for THIS query only -- they still
                       appear as candidates with score 0.  Used to simulate a class-2/3
                       molecule whose true structure has no reference spectra.
        Both exclusions are applied per call, so one library can serve every query.

        Returns a DataFrame (one row per candidate structure, best first) with columns
        ik14, smiles, mass, score, n_lib_spectra, best_cosine, mean_qmax, n_matched_best,
        matched_mode, same_adduct, n_train_spectra.
        """
        cfg = self.config
        rows = self.candidates(mass_lo, mass_hi)
        n_cand = len(rows)
        base = pd.DataFrame({
            'ik14': self.ik14[rows],
            'smiles': self.smiles[rows],
            'mass': self.struct_mass[rows],
            'n_train_spectra': self.n_struct_spectra[rows],
        })
        if n_cand == 0:
            for c, dt in (('score', np.float64), ('n_lib_spectra', np.int64), ('best_cosine', np.float64),
                          ('mean_qmax', np.float64), ('n_matched_best', np.int32),
                          ('matched_mode', bool), ('same_adduct', bool)):
                base[c] = np.zeros(0, dt)
            return base

        spec_ids = self._spectra_of(rows)
        # candidate-structure row -> position in `rows`
        pos_of_struct = np.zeros(len(self.ik14), np.int64)
        pos_of_struct[rows] = np.arange(n_cand)
        if len(exclude_ik14):
            blind = np.isin(self.ik14[rows], np.atleast_1d(exclude_ik14))
            if blind.any():
                spec_ids = spec_ids[~blind[pos_of_struct[self.spec_struct[spec_ids]]]]
        if len(exclude_libs):
            spec_ids = spec_ids[~np.isin(self.spec_lib[spec_ids], np.atleast_1d(exclude_libs))]
        n_lib = len(spec_ids)
        lib_cand = pos_of_struct[self.spec_struct[spec_ids]]
        n_lib_spectra = np.bincount(lib_cand, minlength=n_cand)

        if n_lib == 0 or not query_spectra:
            base['score'] = 0.0
            base['n_lib_spectra'] = n_lib_spectra
            base['best_cosine'] = 0.0
            base['mean_qmax'] = 0.0
            base['n_matched_best'] = 0
            base['matched_mode'] = False
            base['same_adduct'] = False
            return _finalise(base)

        # concatenated, globally m/z-sorted peak block for the whole candidate set
        local = np.arange(n_lib, dtype=np.int64)
        counts = self._coff[spec_ids + 1] - self._coff[spec_ids]
        gather = _ranges(self._coff[spec_ids], counts)
        block = _PeakBlock(self._cmz[gather], self._cw[gather], np.repeat(local, counts))
        lnorm = self._cnorm[spec_ids]
        lmode = self.spec_mode[spec_ids]
        lprec = self.spec_prec[spec_ids]
        ladd = self.spec_adduct[spec_ids]

        n_q = len(query_spectra)
        cos = np.zeros((n_q, n_lib), dtype=np.float32)
        nmatch = np.zeros((n_q, n_lib), dtype=np.int32)
        q_mode_ok = np.zeros((n_q, n_lib), dtype=bool)
        q_add_ok = np.zeros((n_q, n_lib), dtype=bool)

        for qi_, q in enumerate(query_spectra):
            prec = float(q.get('precursor_mz', np.nan))
            qmz, qit = prep_spectrum(q['mzs'], q['intensities'], prec,
                                     max_peaks=cfg.max_peaks, min_rel=cfg.min_rel,
                                     drop_precursor=cfg.drop_precursor)
            if len(qmz) < 2:
                continue
            qw = _transform(qmz, qit, cfg)
            qnorm = np.sqrt((qw * qw).sum())
            if qnorm <= 0:
                continue
            qmode = 1 if _mode_key(q.get('ionization_mode')) == 'positive' else -1
            mode_ok = lmode == qmode
            add_ok = ladd == q.get('adduct')
            q_mode_ok[qi_] = mode_ok
            q_add_ok[qi_] = add_ok
            use = mode_ok if cfg.same_mode_only else np.ones(n_lib, bool)
            if not use.any():
                continue
            if use.all():
                sub_block, sub_norm, sub_map = block, lnorm, None
            else:
                pkeep = use[block.spec]
                remap = np.full(n_lib, -1, np.int64)
                sub_ids = np.flatnonzero(use)
                remap[sub_ids] = np.arange(len(sub_ids))
                sub_block = _PeakBlock(block.mz[pkeep], block.weight[pkeep], remap[block.spec[pkeep]])
                sub_norm = lnorm[sub_ids]
                sub_map = sub_ids
            n_sub = n_lib if sub_map is None else len(sub_map)
            shifts = None
            if cfg.modified and np.isfinite(prec):
                sp = (prec - (lprec if sub_map is None else lprec[sub_map]))
                shifts = np.round(sp, 4)
            num, cnt = _cosine_block(qmz, qw, sub_block, n_sub, cfg, shifts=shifts)
            c = num / (qnorm * np.maximum(sub_norm, 1e-12))
            if cfg.min_matches > 1:
                c = np.where(cnt >= cfg.min_matches, c, 0.0)
            if cfg.same_adduct_bonus > 0:
                sa = add_ok if sub_map is None else add_ok[sub_map]
                c = c * np.where(sa, 1.0, 1.0 - cfg.same_adduct_bonus)
            c = np.clip(c, 0.0, 1.0)
            if sub_map is None:
                cos[qi_] = c
                nmatch[qi_] = cnt
            else:
                cos[qi_, sub_map] = c
                nmatch[qi_, sub_map] = cnt

        return _aggregate(base, cos, nmatch, lib_cand, n_cand, n_lib_spectra,
                          q_mode_ok, q_add_ok, cfg)


def _aggregate(base, cos, nmatch, lib_cand, n_cand, n_lib_spectra, q_mode_ok, q_add_ok, cfg):
    """Collapse the (query spectrum x library spectrum) cosine matrix onto candidate structures."""
    n_q = cos.shape[0]
    # library spectra grouped by candidate structure (contiguous segments)
    order = np.argsort(lib_cand, kind='stable')
    counts = n_lib_spectra.astype(np.int64)
    starts = np.zeros(n_cand, np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    qmax = _segment_max(cos[:, order].astype(np.float64), starts, counts)     # (n_q, n_cand)
    best = qmax.max(axis=0) if n_q else np.zeros(n_cand)

    mo = q_mode_ok[:, order]
    usable = _segment_any(mo, starts, counts) if n_q else np.zeros((0, n_cand), bool)
    if not cfg.same_mode_only and n_q:
        usable = np.repeat((counts > 0)[None, :], n_q, axis=0)

    denom = usable.sum(axis=0) if n_q else np.zeros(n_cand)
    mean_qmax = np.where(denom > 0, (qmax * usable).sum(axis=0) / np.maximum(denom, 1), 0.0) \
        if n_q else np.zeros(n_cand)

    if cfg.agg == 'max':
        score = best
    elif cfg.agg == 'mean_qmax_all':
        score = qmax.mean(axis=0) if n_q else np.zeros(n_cand)
    elif cfg.agg.startswith('mean_top') and n_q:
        k = min(int(cfg.agg[len('mean_top'):]), n_q)
        srt = -np.sort(-qmax, axis=0)
        score = srt[:k].mean(axis=0)
    else:
        score = mean_qmax

    nm_best = (_segment_max(nmatch[:, order], starts, counts).max(axis=0)
               if n_q else np.zeros(n_cand, np.int32))
    matched_mode = usable.any(axis=0) if n_q else np.zeros(n_cand, bool)
    same_adduct = (_segment_any(q_add_ok[:, order], starts, counts).any(axis=0)
                   if n_q else np.zeros(n_cand, bool))

    base = base.copy()
    base['score'] = np.clip(score, 0.0, 1.0)
    base['n_lib_spectra'] = n_lib_spectra
    base['best_cosine'] = best
    base['mean_qmax'] = mean_qmax
    base['n_matched_best'] = nm_best
    base['matched_mode'] = matched_mode
    base['same_adduct'] = same_adduct
    return _finalise(base)


def _finalise(df):
    # ties (notably the many score-0 mass coincidences) are broken by how many spectra the
    # structure has in the library -- a weak but real "this compound gets measured" prior
    df = df.sort_values(['score', 'n_lib_spectra', 'ik14'], ascending=[False, False, True])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# Window helpers
# ─────────────────────────────────────────────────────────────────────

def _merge_windows(windows):
    w = np.asarray([wi for wi in windows if np.isfinite(wi[0]) and np.isfinite(wi[1])], dtype=np.float64)
    if len(w) == 0:
        return np.zeros(0), np.zeros(0)
    w = w[np.argsort(w[:, 0])]
    lo, hi = [w[0, 0]], [w[0, 1]]
    for a, b in w[1:]:
        if a <= hi[-1]:
            hi[-1] = max(hi[-1], b)
        else:
            lo.append(a)
            hi.append(b)
    return np.asarray(lo), np.asarray(hi)


def _in_windows(mass, lo, hi):
    if len(lo) == 0:
        return np.zeros(len(mass), bool)
    idx = np.searchsorted(lo, mass, side='right') - 1
    ok = idx >= 0
    out = np.zeros(len(mass), bool)
    out[ok] = mass[ok] <= hi[idx[ok]]
    return out & np.isfinite(mass)


# ─────────────────────────────────────────────────────────────────────
# Convenience: a whole test/validation set in one call
# ─────────────────────────────────────────────────────────────────────

def molecule_spectra(df, id_col='molecule_id'):
    """Group a test-style DataFrame into {molecule_id: [spectrum dicts]}."""
    out = {}
    for mid, g in df.groupby(id_col, sort=False):
        out[mid] = [{'mzs': r.ms2_mzs, 'intensities': r.ms2_normalized_intensities,
                     'precursor_mz': float(r.precursor_mz), 'adduct': r.adduct,
                     'ionization_mode': r.ionization_mode} for r in g.itertuples(index=False)]
    return out


def rank_molecules(train_path, structures_df, queries, config=None, ppm=DEFAULT_MASS_PPM,
                   exclude_libs=(), max_ppm_error=20.0, top_k=25, verbose=True, lib=None):
    """End-to-end: build the library for all queries, search each, return {mol_id: DataFrame}.

    queries: {molecule_id: [spectrum dicts]} as produced by `molecule_spectra`.
    """
    masses, windows = {}, []
    for mid, specs in queries.items():
        m = estimate_molecule_mass([s['precursor_mz'] for s in specs],
                                   [s['adduct'] for s in specs],
                                   [s['ionization_mode'] for s in specs])
        masses[mid] = m
        if np.isfinite(m):
            windows.append(mass_window(m, ppm))
    if lib is None:
        lib = SpectralLibrary.from_train(train_path, structures_df, windows,
                                         exclude_libs=exclude_libs, max_ppm_error=max_ppm_error,
                                         config=config, verbose=verbose)
    elif config is not None:
        lib.set_config(config)
    out = {}
    for mid, specs in queries.items():
        m = masses[mid]
        if not np.isfinite(m):
            out[mid] = pd.DataFrame(columns=['ik14', 'smiles', 'score'])
            continue
        lo, hi = mass_window(m, ppm)
        out[mid] = lib.search(specs, lo, hi).head(top_k) if top_k else lib.search(specs, lo, hi)
    return out, lib
