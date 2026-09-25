"""
Model inference helpers: raw spectra -> tensors, per-molecule fingerprint prediction, candidate
log-likelihood scoring and de novo sampling.
"""

import numpy as np
import torch

from casmi.common import (
    ADDUCT_TO_ID, MAX_PEAKS, PAD_ID, SmilesTokenizer, prep_peaks, first_collision_energy,
)
from casmi.model import load_model


def spectra_to_batch(spectra, device):
    """spectra: list of dicts with mzs, intensities, precursor_mz, adduct, collision_energy_ev (optional)."""
    n = len(spectra)
    mz = np.zeros((n, MAX_PEAKS), np.float32)
    inten = np.zeros((n, MAX_PEAKS), np.float32)
    prec = np.zeros(n, np.float64)
    adduct = np.zeros(n, np.int64)
    ce = np.full(n, -1.0, np.float32)
    for i, sp in enumerate(spectra):
        m, it = prep_peaks(sp['mzs'], sp['intensities'], sp['precursor_mz'])
        mz[i, :len(m)], inten[i, :len(m)] = m, it
        prec[i] = sp['precursor_mz']
        adduct[i] = ADDUCT_TO_ID.get(sp.get('adduct'), 0)
        c = first_collision_energy(sp.get('collision_energy_ev'))
        ce[i] = c if np.isfinite(c) else -1.0
    batch = dict(mz=mz, inten=inten, prec=prec, adduct=adduct, ce=ce)
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


class ModelScorer:
    def __init__(self, model_path, vocab_path=None, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model, self.ckpt = load_model(model_path, self.device)
        if self.ckpt.get('cfg', {}).get('vocab'):
            self.tokenizer = SmilesTokenizer(self.ckpt['cfg']['vocab'])
        elif vocab_path and os.path.exists(vocab_path):
            self.tokenizer = SmilesTokenizer.load(vocab_path)
        else:
            raise ValueError(f"No vocabulary found in checkpoint or at {vocab_path}")
        # bf16 only where the hardware really has it (Ampere+); a Kaggle T4 runs fp32, which is fast enough here
        self.amp = self.device.type == 'cuda' and torch.cuda.get_device_capability(self.device)[0] >= 8
        self.max_tokens = self.model.decoder.max_tokens

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    @torch.inference_mode()
    def encode(self, spectra):
        with self._autocast():
            return self.model.encode(spectra_to_batch(spectra, self.device))

    @torch.inference_mode()
    def fingerprint(self, states, mask):
        """Mean predicted fingerprint probability over a molecule's spectra -> numpy [4096]."""
        with self._autocast():
            return self.model.predict_fingerprint(states, mask).mean(0).cpu().numpy()

    def tokenize(self, smiles_list):
        """-> (token matrix [N,L], ok mask). SMILES that cannot be tokenised or are too long get ok=False."""
        ids = [self.tokenizer.encode(s) if isinstance(s, str) else None for s in smiles_list]
        ok = np.array([t is not None and len(t) <= self.max_tokens for t in ids])
        width = max([len(t) for t, good in zip(ids, ok) if good], default=2)
        mat = np.full((len(ids), width), PAD_ID, np.int64)
        for i, (t, good) in enumerate(zip(ids, ok)):
            if good:
                mat[i, :len(t)] = t
        return mat, ok

    @torch.inference_mode()
    def loglik(self, states, mask, smiles_list, max_rows=768):
        """Mean over the molecule's spectra of log p(SMILES | spectrum). Returns (loglik [N], n_tokens [N]); -inf if unscorable."""
        n_spec = states.shape[0]
        mat, ok = self.tokenize(smiles_list)
        out = np.full(len(smiles_list), -np.inf, np.float64)
        n_tok = (mat != PAD_ID).sum(1) - 1
        good = np.flatnonzero(ok)
        order = good[np.argsort(n_tok[good])]                  # similar lengths together -> little padding
        chunk = max(1, max_rows // n_spec)
        for i in range(0, len(order), chunk):
            sel = order[i:i + chunk]
            width = int(n_tok[sel].max()) + 1
            toks = torch.from_numpy(mat[sel, :width]).to(self.device)
            c = len(sel)
            st = states.repeat(c, 1, 1)                        # [c*S, P, D] : candidate-major blocks of S spectra
            mk = mask.repeat(c, 1)
            tk = toks.repeat_interleave(n_spec, dim=0)
            with self._autocast():
                lp, _ = self.model.score(st, mk, tk)
            out[sel] = lp.view(c, n_spec).mean(1).double().cpu().numpy()
        return out, n_tok

    @torch.inference_mode()
    def sample(self, states, mask, n_samples=64, temperature=1.0, top_k=0, max_new_tokens=160, max_rows=1024):
        """De novo sampling for every spectrum. Returns list of (smiles, logprob) over all spectra (finished samples only)."""
        results = []
        per_call = max(1, max_rows // n_samples)
        for i in range(0, states.shape[0], per_call):
            with self._autocast():
                tokens, logprob = self.model.generate(states[i:i + per_call], mask[i:i + per_call], n_samples=n_samples,
                                                      temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens)
            tokens, logprob = tokens.cpu().numpy(), logprob.float().cpu().numpy()
            for b in range(tokens.shape[0]):
                for j in range(tokens.shape[1]):
                    if np.isfinite(logprob[b, j]):
                        results.append((self.tokenizer.decode(tokens[b, j, 1:]), float(logprob[b, j])))
        return results
