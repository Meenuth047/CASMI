"""
Spectrum -> structure model.

  SpectrumEncoder : peaks (m/z, neutral loss, intensity) + a metadata token (precursor m/z, neutral mass,
                    adduct, collision energy) -> contextual peak states
  FingerprintHead : pooled encoder state -> 4096-bit Morgan fingerprint logits   (ranks database candidates)
  SmilesDecoder   : autoregressive SMILES decoder with cross-attention           (de novo + candidate likelihood)

Differences from the tutorial model that matter for correctness:
  * teacher forcing shifts the labels: position i is trained to predict token i+1 (the tutorial trained
    position i to predict token i, which a causal decoder solves by copying its input);
  * generation feeds the whole prefix through a causal decoder (with a KV cache), not just the last token.
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from casmi.common import PAD_ID, BOS_ID, EOS_ID, FP_BITS, TEST_ADDUCTS, parse_adduct

N_ADDUCTS = len(TEST_ADDUCTS) + 1   # 0 = unknown

# adduct id -> (n_mol, shift, |charge|) so the model can be told the implied neutral mass
_ADDUCT_TABLE = np.zeros((N_ADDUCTS, 3), dtype=np.float64)
_ADDUCT_TABLE[0] = (1, 0.0, 1)
for _i, _a in enumerate(TEST_ADDUCTS):
    _n, _shift, _z = parse_adduct(_a)
    _ADDUCT_TABLE[_i + 1] = (_n, _shift, abs(_z))


def default_config(vocab_size):
    return dict(vocab_size=vocab_size, d_model=512, n_heads=8, enc_layers=6, dec_layers=6, d_ff=2048,
                dropout=0.1, max_tokens=192, fp_bits=FP_BITS)


class SinusoidalMz(nn.Module):
    """Fixed sinusoidal features of a mass value, wavelengths log-spaced from 1e-2 to 1e3 Da."""

    def __init__(self, dim, min_wavelength=1e-2, max_wavelength=1e3):
        super().__init__()
        wavelengths = torch.logspace(math.log10(min_wavelength), math.log10(max_wavelength), dim // 2)
        self.register_buffer('freq', 2 * math.pi / wavelengths, persistent=False)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            phase = x.float().unsqueeze(-1) * self.freq
            return torch.cat([phase.sin(), phase.cos()], dim=-1)


class Block(nn.Module):
    """Pre-LN transformer block: self-attention, optional cross-attention, MLP."""

    def __init__(self, d_model, n_heads, d_ff, dropout, cross=False):
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.cross = cross
        if cross:
            self.ln_x = nn.LayerNorm(d_model)
            self.q_x = nn.Linear(d_model, d_model)
            self.kv_x = nn.Linear(d_model, 2 * d_model)
            self.proj_x = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def _heads(self, x):
        b, t, d = x.shape
        return x.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)

    def _merge(self, x):
        b, h, t, dh = x.shape
        return x.transpose(1, 2).reshape(b, t, h * dh)

    def memory_kv(self, memory):
        k, v = self.kv_x(memory).chunk(2, dim=-1)
        return self._heads(k), self._heads(v)

    def forward(self, x, attn_mask=None, causal=False, memory_kv=None, memory_mask=None, cache=None):
        """attn_mask / memory_mask: bool [B,1,1,S], True = attend. cache: dict with 'k','v' for incremental decoding."""
        p = self.dropout if self.training else 0.0
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        if cache is not None:
            if 'k' in cache:
                k = torch.cat([cache['k'], k], dim=2)
                v = torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
            # a single new query attends to every cached position, so no causal mask is needed
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p,
                                               is_causal=causal and attn_mask is None)
        x = x + self.drop(self.proj(self._merge(y)))
        if self.cross:
            qx = self._heads(self.q_x(self.ln_x(x)))
            kx, vx = memory_kv
            y = F.scaled_dot_product_attention(qx, kx, vx, attn_mask=memory_mask, dropout_p=p)
            x = x + self.drop(self.proj_x(self._merge(y)))
        return x + self.drop(self.mlp(self.ln2(x)))


class SpectrumEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.sin = SinusoidalMz(d)
        self.peak_mlp = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, d))
        self.adduct_emb = nn.Embedding(N_ADDUCTS, 64)
        self.meta_mlp = nn.Sequential(nn.Linear(2 * d + 64 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.register_buffer('adduct_table', torch.tensor(_ADDUCT_TABLE), persistent=False)
        self.blocks = nn.ModuleList([Block(d, cfg['n_heads'], cfg['d_ff'], cfg['dropout']) for _ in range(cfg['enc_layers'])])
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg['dropout'])

    def neutral_mass(self, prec, adduct):
        tab = self.adduct_table[adduct]
        return (prec.double() * tab[:, 2] - tab[:, 1]) / tab[:, 0]

    def forward(self, mz, inten, prec, adduct, ce):
        """mz/inten [B,P] (a slot with intensity 0 is padding), prec [B] float64, adduct [B] long, ce [B] (-1 = unknown).

        Returns (states [B,1+P,D], mask [B,1+P] bool) — position 0 is the metadata token.
        """
        b, p = mz.shape
        peak_mask = inten > 0
        loss = (prec[:, None] - mz.double()).clamp(min=0).float()
        inten = inten.float()
        peaks = self.peak_mlp(torch.cat([self.sin(mz), self.sin(loss), inten[..., None], inten.sqrt()[..., None]], dim=-1))
        has_ce = (ce >= 0).float()
        meta = self.meta_mlp(torch.cat([
            self.sin(prec.float()), self.sin(self.neutral_mass(prec, adduct).float()), self.adduct_emb(adduct),
            (ce.clamp(min=0) / 100.0)[:, None], has_ce[:, None]], dim=-1))
        x = self.drop(torch.cat([meta[:, None, :], peaks], dim=1))
        mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mz.device), peak_mask], dim=1)
        attn_mask = mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return self.ln(x), mask


class FingerprintHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.net = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(cfg['dropout']), nn.Linear(2 * d, cfg['fp_bits']))

    def forward(self, states, mask):
        m = mask[..., None].to(states.dtype)
        mean = (states * m).sum(1) / m.sum(1).clamp(min=1)
        return self.net(torch.cat([states[:, 0], mean], dim=-1))


class SmilesDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['d_model']
        self.max_tokens = cfg['max_tokens']
        self.tok = nn.Embedding(cfg['vocab_size'], d, padding_idx=PAD_ID)
        self.pos = nn.Embedding(cfg['max_tokens'], d)
        self.blocks = nn.ModuleList([Block(d, cfg['n_heads'], cfg['d_ff'], cfg['dropout'], cross=True) for _ in range(cfg['dec_layers'])])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg['vocab_size'], bias=False)
        self.drop = nn.Dropout(cfg['dropout'])

    def forward(self, tokens_in, memory, memory_mask):
        """Teacher forcing. tokens_in [B,T] = sequence WITHOUT its last token; returns logits [B,T,V] for tokens 1..T."""
        t = tokens_in.shape[1]
        x = self.drop(self.tok(tokens_in) + self.pos(torch.arange(t, device=tokens_in.device))[None])
        mem_mask = memory_mask[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, causal=True, memory_kv=blk.memory_kv(memory), memory_mask=mem_mask)
        return self.head(self.ln(x))

    def step(self, token, position, memory_kvs, mem_mask, caches):
        x = self.tok(token) + self.pos(torch.full_like(token, position))
        for blk, mkv, cache in zip(self.blocks, memory_kvs, caches):
            x = blk(x, memory_kv=mkv, memory_mask=mem_mask, cache=cache)
        return self.head(self.ln(x))[:, -1]


class CasmiModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.encoder = SpectrumEncoder(cfg)
        self.fp_head = FingerprintHead(cfg)
        self.decoder = SmilesDecoder(cfg)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode(self, batch):
        return self.encoder(batch['mz'], batch['inten'], batch['prec'], batch['adduct'], batch['ce'])

    def forward(self, batch):
        """Training forward. batch['tokens'] is [B,L] = BOS ... EOS PAD*. Returns dict of losses."""
        states, mask = self.encode(batch)
        tokens = batch['tokens']
        logits = self.decoder(tokens[:, :-1], states, mask)           # inputs : BOS t1 ... t(n-1)
        targets = tokens[:, 1:]                                        # targets: t1 ... EOS   <- the shift
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD_ID)
        out = {'ce': ce}
        if 'fp' in batch:
            fp_logits = self.fp_head(states, mask).float()
            out['bce'] = F.binary_cross_entropy_with_logits(fp_logits, batch['fp'])
            out['fp_cos'] = F.cosine_similarity(fp_logits.sigmoid(), batch['fp'], dim=-1).mean()
        return out

    @torch.inference_mode()
    def predict_fingerprint(self, states, mask):
        return self.fp_head(states, mask).float().sigmoid()

    @torch.inference_mode()
    def score(self, states, mask, tokens):
        """Sum log p(tokens | spectrum) for already-paired rows: states [N,S,D], tokens [N,L] (BOS..EOS PAD*)."""
        logits = self.decoder(tokens[:, :-1], states, mask).float()
        logp = logits.log_softmax(-1).gather(-1, tokens[:, 1:, None]).squeeze(-1)
        keep = tokens[:, 1:] != PAD_ID
        return (logp * keep).sum(-1), keep.sum(-1)

    @torch.inference_mode()
    def generate(self, states, mask, n_samples=25, max_new_tokens=160, temperature=1.0, top_k=0):
        """Ancestral sampling with a KV cache. Returns (tokens [B,n,L], logprob [B,n]) — logprob is under T=1."""
        b, s, d = states.shape
        n = b * n_samples
        states = states.repeat_interleave(n_samples, dim=0)
        mem_mask = mask.repeat_interleave(n_samples, dim=0)[:, None, None, :]
        memory_kvs = [blk.memory_kv(states) for blk in self.decoder.blocks]
        caches = [dict() for _ in self.decoder.blocks]
        token = torch.full((n, 1), BOS_ID, dtype=torch.long, device=states.device)
        seqs, logprob = [token], torch.zeros(n, device=states.device)
        done = torch.zeros(n, dtype=torch.bool, device=states.device)
        max_new_tokens = min(max_new_tokens, self.decoder.max_tokens - 1)
        for position in range(max_new_tokens):
            logits = self.decoder.step(token, position, memory_kvs, mem_mask, caches).float()
            logp = logits.log_softmax(-1)
            sample_logits = logits / max(temperature, 1e-6)
            if top_k and top_k < sample_logits.size(-1):
                kth = sample_logits.topk(top_k, dim=-1).values[:, -1:]
                sample_logits = sample_logits.masked_fill(sample_logits < kth, float('-inf'))
            nxt = torch.multinomial(sample_logits.softmax(-1), 1)
            nxt = torch.where(done[:, None], torch.full_like(nxt, PAD_ID), nxt)
            logprob = logprob + torch.where(done, torch.zeros_like(logprob), logp.gather(-1, nxt).squeeze(-1))
            seqs.append(nxt)
            done = done | (nxt.squeeze(-1) == EOS_ID)
            token = nxt
            if bool(done.all()):
                break
        tokens = torch.cat(seqs, dim=1)
        logprob = torch.where(done, logprob, torch.full_like(logprob, float('-inf')))   # unfinished = invalid
        return tokens.view(b, n_samples, -1), logprob.view(b, n_samples)


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = CasmiModel(ckpt['cfg'])
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval(), ckpt
