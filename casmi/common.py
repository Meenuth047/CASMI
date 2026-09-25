"""
Shared core for the CASMI 2026 pipeline: adduct arithmetic, the Kaggle metric key,
spectrum preprocessing, SMILES tokenisation and fingerprints.

Pure python + numpy + rdkit so the same file runs locally and inside the Kaggle notebook.
"""

import re
import json
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import rdkit.RDLogger as _rkl
import rdkit.rdBase as _rkrb

_rkl.logger().setLevel(_rkl.ERROR)
_rkrb.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────
# Adduct arithmetic
# ─────────────────────────────────────────────────────────────────────

ELECTRON_MASS = 0.00054857990907
_PT = Chem.GetPeriodicTable()

# The ten adducts the competition says the hidden test set uses.
TEST_ADDUCTS = [
    '[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+',
    '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-',
]
ADDUCT_TO_ID = {a: i + 1 for i, a in enumerate(TEST_ADDUCTS)}  # 0 = unknown/other

_FORMULA_TOKEN = re.compile(r'([A-Z][a-z]?)(\d*)')
_ADDUCT_RE = re.compile(r'^\[(\d*)M([^\]]*)\](\d*)([+-])$')
_ADDUCT_PART = re.compile(r'([+-])(\d*)([A-Za-z][A-Za-z0-9]*)')


def formula_mass(formula):
    """Monoisotopic mass of a plain formula string such as 'CH2O2'."""
    total = 0.0
    for sym, count in _FORMULA_TOKEN.findall(formula):
        total += _PT.GetMostCommonIsotopeMass(sym) * (int(count) if count else 1)
    return total


_adduct_cache = {}


def parse_adduct(adduct):
    """'[2M+Na-2H]-' -> (n_mol, mass_shift, charge) with m/z = (n_mol*M + mass_shift) / |charge|.

    The shift includes the electron-mass correction. Returns None when the string cannot be parsed.
    """
    if adduct in _adduct_cache:
        return _adduct_cache[adduct]
    result = None
    m = _ADDUCT_RE.match(adduct.replace(' ', '')) if isinstance(adduct, str) else None
    if m:
        n_mol = int(m.group(1)) if m.group(1) else 1
        charge = (int(m.group(3)) if m.group(3) else 1) * (1 if m.group(4) == '+' else -1)
        body, shift, consumed = m.group(2), 0.0, 0
        try:
            for sign, mult, formula in _ADDUCT_PART.findall(body):
                shift += (1 if sign == '+' else -1) * (int(mult) if mult else 1) * formula_mass(formula)
                consumed += len(sign) + len(mult) + len(formula)
            if consumed == len(body):
                result = (n_mol, shift - charge * ELECTRON_MASS, charge)
        except Exception:
            result = None
    _adduct_cache[adduct] = result
    return result


def neutral_mass_from_mz(precursor_mz, adduct):
    """Monoisotopic neutral mass implied by a precursor m/z and adduct; NaN if the adduct is unknown."""
    parsed = parse_adduct(adduct)
    if parsed is None:
        return float('nan')
    n_mol, shift, charge = parsed
    return (precursor_mz * abs(charge) - shift) / n_mol


def mz_from_neutral_mass(mass, adduct):
    parsed = parse_adduct(adduct)
    if parsed is None:
        return float('nan')
    n_mol, shift, charge = parsed
    return (n_mol * mass + shift) / abs(charge)


# Measured on both timsTOF libraries in train (enveda-180, enveda-np-examples): the neutral mass implied by the
# recorded precursor m/z sits about +2.0 ppm (positive mode) / +0.6 ppm (negative mode) above the true mass.
MASS_BIAS_PPM = {1: 2.0, -1: 0.6}


def molecule_neutral_mass(precursor_mzs, adducts, calibrate=True):
    """One neutral-mass estimate for a molecule from all of its spectra (median is robust to a bad adduct)."""
    masses = []
    for mz, ad in zip(precursor_mzs, adducts):
        m = neutral_mass_from_mz(mz, ad)
        if np.isfinite(m):
            if calibrate:
                m *= 1.0 - MASS_BIAS_PPM[1 if parse_adduct(ad)[2] > 0 else -1] * 1e-6
            masses.append(m)
    return float(np.median(masses)) if masses else float('nan')


# ─────────────────────────────────────────────────────────────────────
# Structure keys — the Kaggle metric compares tautomer-canonical InChIKey first blocks
# ─────────────────────────────────────────────────────────────────────

_taut_enum = None


def _tautomer_enumerator():
    global _taut_enum
    if _taut_enum is None:
        _taut_enum = rdMolStandardize.TautomerEnumerator()
    return _taut_enum


def mol_from_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def plain_key(smiles_or_mol):
    """InChIKey first block with no tautomer canonicalisation — cheap, for bulk dedup."""
    mol = mol_from_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
        return key.split('-')[0] if key else None
    except Exception:
        return None


def metric_key(smiles_or_mol):
    """The key the competition metric compares: tautomer-canonicalise, then InChIKey first block.

    Falls back to the plain key if canonicalisation fails, and None if the SMILES is invalid.
    """
    mol = mol_from_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if mol is None:
        return None
    try:
        canon = _tautomer_enumerator().Canonicalize(mol)
        key = Chem.MolToInchiKey(canon)
        if key:
            return key.split('-')[0]
    except Exception:
        pass
    return plain_key(mol)


def flat_canonical_smiles(smiles_or_mol):
    """Canonical SMILES with stereochemistry removed (the metric ignores it)."""
    mol = mol_from_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


def exact_mass(smiles_or_mol):
    mol = mol_from_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if mol is None:
        return float('nan')
    # ExactMolWt ignores electrons; correct for formal charge so ions compare properly.
    return ExactMolWt(mol) - Chem.GetFormalCharge(mol) * ELECTRON_MASS


def mol_formula(smiles_or_mol):
    mol = mol_from_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return CalcMolFormula(mol) if mol is not None else None


def ppm_window(mass, ppm):
    delta = mass * ppm * 1e-6
    return mass - delta, mass + delta


# ─────────────────────────────────────────────────────────────────────
# Fingerprints
# ─────────────────────────────────────────────────────────────────────

FP_BITS = 4096
_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)


def morgan_bits(smiles_or_mol):
    """Indices of the on-bits of the 4096-bit Morgan radius-2 fingerprint (None if invalid)."""
    mol = mol_from_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    if mol is None:
        return None
    return np.fromiter(_fp_gen.GetFingerprint(mol).GetOnBits(), dtype=np.int16)


def bits_to_dense(bit_lists, n_bits=FP_BITS, dtype=np.float32):
    out = np.zeros((len(bit_lists), n_bits), dtype=dtype)
    for i, bits in enumerate(bit_lists):
        if bits is not None and len(bits):
            out[i, bits] = 1
    return out


# ─────────────────────────────────────────────────────────────────────
# Spectrum preprocessing
# ─────────────────────────────────────────────────────────────────────

MAX_PEAKS = 128


def prep_peaks(mzs, intensities, precursor_mz, max_peaks=MAX_PEAKS, min_rel_intensity=0.0):
    """Drop peaks above precursor+2 Da, keep the `max_peaks` most intense, renormalise to base peak 1.

    Returns (mzs, intensities) as float32 arrays sorted by ascending m/z.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    keep = (mzs <= precursor_mz + 2.0) & (intensities > 0) & np.isfinite(mzs) & np.isfinite(intensities)
    mzs, intensities = mzs[keep], intensities[keep]
    if len(mzs) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    intensities = intensities / intensities.max()
    if min_rel_intensity > 0:
        keep = intensities >= min_rel_intensity
        mzs, intensities = mzs[keep], intensities[keep]
    if len(mzs) > max_peaks:
        top = np.argpartition(intensities, -max_peaks)[-max_peaks:]
        mzs, intensities = mzs[top], intensities[top]
    order = np.argsort(mzs)
    return mzs[order].astype(np.float32), intensities[order].astype(np.float32)


def first_collision_energy(value):
    """collision_energy_ev is a list (or null); summarise as its mean, NaN when absent."""
    if value is None:
        return float('nan')
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if len(arr) else float('nan')
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────────────────────────────
# SMILES tokeniser (atom level — no external tokenizer dependency)
# ─────────────────────────────────────────────────────────────────────

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
_SPECIALS = ['<pad>', '<s>', '</s>', '<unk>']
_SMILES_TOKEN = re.compile(
    r'(\[[^\]]+\]|Br|Cl|Si|Se|se|@@|%\d{2}|[BCNOPSFIbcnops]|[()=#\-+\\/:~.*$]|\d)'
)


def split_smiles(smiles):
    return _SMILES_TOKEN.findall(smiles)


class SmilesTokenizer:
    def __init__(self, vocab):
        self.itos = list(vocab)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(cls, smiles_iter, min_count=1):
        counts = {}
        for smi in smiles_iter:
            for tok in split_smiles(smi):
                counts[tok] = counts.get(tok, 0) + 1
        toks = sorted(t for t, c in counts.items() if c >= min_count)
        return cls(_SPECIALS + toks)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def save(self, path):
        with open(path, 'w') as fh:
            json.dump(self.itos, fh)

    def __len__(self):
        return len(self.itos)

    def encode(self, smiles):
        """-> [BOS, tokens..., EOS]; None if the SMILES contains an out-of-vocabulary token."""
        toks = split_smiles(smiles)
        if ''.join(toks) != smiles:
            return None
        ids = [self.stoi.get(t, UNK_ID) for t in toks]
        if UNK_ID in ids:
            return None
        return [BOS_ID] + ids + [EOS_ID]

    def decode(self, ids):
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else '')
        return ''.join(out)
