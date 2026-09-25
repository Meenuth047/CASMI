"""
Train the V2 spectrum -> (fingerprint, SMILES) model.

Key improvements over V1:
  1. Randomized SMILES targets (4 variants per structure; 99.4% differ from canonical)
     so string memorisation is eliminated.
  2. Structure-only pre-training from COCONUT (~25% of each batch) with mass/adduct conditioning
     to teach natural-product chemistry and mass awareness.
  3. Balanced sampling capping spectra per structure to prevent overfitting to common molecules.
  4. Higher dropout (0.1 -> 0.2).
  5. Expanded 74-token vocabulary (work/v2/vocab.json) stored in checkpoint cfg.
  6. Tracks validation CE on unseen structures and saves the best checkpoint.

Usage:
    ~/casmi-gpu-venv/bin/python -m casmi.train_v2 --epochs 8
    ~/casmi-gpu-venv/bin/python -m casmi.train_v2 --smoke
"""

import os
import glob
import time
import math
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from casmi.common import (
    SmilesTokenizer, PAD_ID, BOS_ID, EOS_ID, FP_BITS, plain_key,
    TEST_ADDUCTS, parse_adduct,
)
from casmi.model import CasmiModel, default_config, _ADDUCT_TABLE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')
V2_DIR = os.path.join(WORK, 'v2')
CKPT_DIR = os.path.join(WORK, 'ckpt')
LIB_E180, LIB_NP = 0, 9


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


# ─────────────────────────────────────────────────────────────────────
# Data Loader
# ─────────────────────────────────────────────────────────────────────

class SpectraDataV2:
    """V2 dataset: spectra with randomized SMILES + structure-only COCONUT."""

    def __init__(self, max_shards=None, smoke=False):
        structs = pd.read_parquet(os.path.join(WORK, 'structures.parquet'))
        self.structs = structs
        self.tokenizer = SmilesTokenizer.load(os.path.join(V2_DIR, 'vocab.json'))
        n_structs = len(structs)

        # Load V2 randomized tokens for training structures: shape (n_structs, 5, 160)
        self.train_tokens = np.load(os.path.join(V2_DIR, 'train_tokens.npy'), mmap_mode='r')
        self.train_ok = np.load(os.path.join(V2_DIR, 'train_ok.npy'))

        # Structure fingerprints
        max_bits = int(structs.fp_bits.map(lambda b: len(b) if b is not None else 0).max())
        self.fp_bits = np.full((n_structs, max_bits), -1, np.int64)
        for i, b in enumerate(structs.fp_bits):
            if b is not None:
                self.fp_bits[i, :len(b)] = b

        self.split = structs.split.values
        self.ik14 = structs.ik14.values

        # Load COCONUT structure-only data
        self.coco_tokens = np.load(os.path.join(V2_DIR, 'coco_tokens.npy'), mmap_mode='r')
        self.coco_mass = np.load(os.path.join(V2_DIR, 'coco_mass.npy'), mmap_mode='r')
        self.coco_fp = np.load(os.path.join(V2_DIR, 'coco_fp.npy'), mmap_mode='r')
        self.n_coco = len(self.coco_mass)
        log(f'loaded COCONUT structure-only: {self.n_coco:,} molecules')

        # Load spectra
        paths = sorted(glob.glob(os.path.join(WORK, 'spectra', 'shard_*.npz')))
        if max_shards:
            paths = paths[-max_shards:]
        parts = [np.load(p) for p in paths]
        cat = lambda k: np.concatenate([p[k] for p in parts])
        self.mz, self.inten, self.prec = cat('mz'), cat('inten'), cat('prec')
        self.adduct, self.ce, self.lib, self.sidx = cat('adduct'), cat('ce'), cat('lib'), cat('sidx')

        spec_split = self.split[self.sidx]
        train_eligible = (spec_split == 'train') & self.train_ok[self.sidx]
        self.train_idx = np.flatnonzero(train_eligible)
        self.val_np_idx = np.flatnonzero((spec_split == 'val_np') & (self.lib == LIB_NP) & self.train_ok[self.sidx])
        self.val_rand_idx = np.flatnonzero((spec_split == 'val_rand') & self.train_ok[self.sidx])

        # Precompute spectrum indices grouped by structure for balanced sampling
        log('indexing spectra per structure...')
        self.struct_to_specs = {}
        for idx in self.train_idx:
            s = self.sidx[idx]
            if s not in self.struct_to_specs:
                self.struct_to_specs[s] = []
            self.struct_to_specs[s].append(idx)
        for s in self.struct_to_specs:
            self.struct_to_specs[s] = np.array(self.struct_to_specs[s], dtype=np.int32)

        log(f'spectra {len(self.sidx):,} | train {len(self.train_idx):,} across {len(self.struct_to_specs):,} structures '
            f'| val_np {len(self.val_np_idx):,} | val_rand {len(self.val_rand_idx):,} | vocab {len(self.tokenizer)}')

    def epoch_indices(self, rng, max_per_struct=6, e180_fraction=0.4, np_upsample=8):
        """Balanced epoch: cap spectra per structure, upsample NP examples."""
        selected = []
        for s, specs in self.struct_to_specs.items():
            if len(specs) <= max_per_struct:
                selected.append(specs)
            else:
                selected.append(rng.choice(specs, size=max_per_struct, replace=False))
        idx = np.concatenate(selected)

        # NP upsampling
        np_specs = idx[self.lib[idx] == LIB_NP]
        if len(np_specs) > 0 and np_upsample > 1:
            idx = np.concatenate([idx, np.repeat(np_specs, np_upsample - 1)])

        rng.shuffle(idx)
        return idx

    def train_batch(self, spec_idx, device, rng, coco_fraction=0.25):
        """Mixed batch: (1-coco_fraction) spectra + (coco_fraction) COCONUT structures."""
        total_b = len(spec_idx)
        n_coco = int(total_b * coco_fraction)
        n_spec = total_b - n_coco

        s_idx = spec_idx[:n_spec]
        s_struct = self.sidx[s_idx]

        # 1. Spectra portion
        spec_mz = self.mz[s_idx]
        spec_inten = self.inten[s_idx].astype(np.float32)
        spec_prec = self.prec[s_idx]
        spec_adduct = self.adduct[s_idx].astype(np.int64)
        spec_ce = self.ce[s_idx]

        # Pick random target variant (1-4, or 0 canonical with 10% prob)
        v_choices = rng.integers(1, 5, size=n_spec)
        use_canon = rng.random(size=n_spec) < 0.10
        v_choices[use_canon] = 0
        spec_toks = self.train_tokens[s_struct, v_choices, :]
        spec_fp = self.fp_bits[s_struct]

        # 2. COCONUT structure-only portion
        c_idx = rng.choice(self.n_coco, size=n_coco, replace=False)
        c_mass = self.coco_mass[c_idx]
        # Random common adduct id: 1 to 10
        c_adduct = rng.integers(1, 11, size=n_coco, dtype=np.int64)
        tab = _ADDUCT_TABLE[c_adduct]
        c_prec = (c_mass * tab[:, 0] + tab[:, 1]) / tab[:, 2]
        c_ce = np.full(n_coco, -1.0, dtype=np.float32)
        c_mz = np.zeros((n_coco, spec_mz.shape[1]), dtype=np.float32)
        c_inten = np.zeros((n_coco, spec_inten.shape[1]), dtype=np.float32)

        c_v_choices = rng.integers(0, 3, size=n_coco)
        coco_toks = self.coco_tokens[c_idx, c_v_choices, :]
        coco_fp = self.coco_fp[c_idx].astype(np.int64)

        # 3. Concatenate
        mz = np.concatenate([spec_mz, c_mz], axis=0)
        inten = np.concatenate([spec_inten, c_inten], axis=0)
        prec = np.concatenate([spec_prec, c_prec], axis=0)
        adduct = np.concatenate([spec_adduct, c_adduct], axis=0)
        ce = np.concatenate([spec_ce, c_ce], axis=0)
        raw_toks = np.concatenate([spec_toks, coco_toks], axis=0)

        # Truncate tokens to max active length in this batch
        tok_lens = (raw_toks != PAD_ID).sum(axis=1)
        max_l = max(int(tok_lens.max()), 2)
        tokens = raw_toks[:, :max_l]

        # Build dense FP targets
        fp_tensor = torch.zeros(total_b, FP_BITS + 1, device=device)
        spec_bits_t = torch.from_numpy(spec_fp).to(device)
        fp_tensor[:n_spec].scatter_(1, spec_bits_t + 1, 1.0)

        coco_bits_t = torch.from_numpy(coco_fp).to(device)
        fp_tensor[n_spec:].scatter_(1, coco_bits_t + 1, 1.0)
        dense_fp = fp_tensor[:, 1:]

        batch = {
            'mz': torch.from_numpy(mz).to(device, non_blocking=True),
            'inten': torch.from_numpy(inten).to(device, non_blocking=True),
            'prec': torch.from_numpy(prec).to(device, non_blocking=True),
            'adduct': torch.from_numpy(adduct).to(device, non_blocking=True),
            'ce': torch.from_numpy(ce).to(device, non_blocking=True),
            'tokens': torch.from_numpy(tokens.astype(np.int64)).to(device, non_blocking=True),
            'fp': dense_fp,
        }
        return batch

    def eval_batch(self, idx, device, with_targets=True):
        """Evaluation batch: canonical SMILES targets (slot 0), no COCONUT, no noise."""
        s = self.sidx[idx]
        out = {
            'mz': torch.from_numpy(self.mz[idx]),
            'inten': torch.from_numpy(self.inten[idx].astype(np.float32)),
            'prec': torch.from_numpy(self.prec[idx]),
            'adduct': torch.from_numpy(self.adduct[idx].astype(np.int64)),
            'ce': torch.from_numpy(self.ce[idx]),
        }
        if with_targets:
            canon_toks = self.train_tokens[s, 0, :]
            tok_lens = (canon_toks != PAD_ID).sum(axis=1)
            max_l = max(int(tok_lens.max()), 2)
            out['tokens'] = torch.from_numpy(canon_toks[:, :max_l].astype(np.int64))

            bits = self.fp_bits[s]
            fp = torch.zeros(len(idx), FP_BITS + 1)
            fp.scatter_(1, torch.from_numpy(bits) + 1, 1.0)
            out['fp'] = fp[:, 1:]

        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
        return out, s


# ─────────────────────────────────────────────────────────────────────
# Augmentation & Evaluation
# ─────────────────────────────────────────────────────────────────────

def augment_peaks(batch, rng_gen):
    """Peak jitter, dropping, and noise insertion for spectrum rows (leaves zero rows untouched)."""
    mz, inten, prec = batch['mz'], batch['inten'], batch['prec']
    b, p = mz.shape
    has_peaks = inten.max(dim=1).values > 0
    if not has_peaks.any():
        return batch

    real = inten > 0
    base = inten >= inten.max(dim=1, keepdim=True).values
    drop_p = torch.rand(b, 1, device=mz.device, generator=rng_gen) * 0.3
    keep = (torch.rand(b, p, device=mz.device, generator=rng_gen) > drop_p) | base
    inten = inten * (real & keep)
    jitter = torch.exp(torch.randn(b, p, device=mz.device, generator=rng_gen) * 0.3)
    inten = inten * jitter

    empty = (inten <= 0) & has_peaks[:, None]
    noisy_row = torch.rand(b, 1, device=mz.device, generator=rng_gen) < 0.5
    fill_p = torch.rand(b, 1, device=mz.device, generator=rng_gen) * 0.3
    add = empty & noisy_row & (torch.rand(b, p, device=mz.device, generator=rng_gen) < fill_p)
    noise_mz = 50.0 + torch.rand(b, p, device=mz.device, generator=rng_gen) * (prec[:, None].float() - 50.0).clamp(min=1.0)
    noise_int = 0.002 + torch.rand(b, p, device=mz.device, generator=rng_gen) ** 2 * 0.05
    mz = torch.where(add, noise_mz, mz)
    inten = torch.where(add, noise_int, inten)

    max_val = inten.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    inten = torch.where(has_peaks[:, None], inten / max_val, inten)
    batch['mz'], batch['inten'] = mz, inten
    return batch


@torch.no_grad()
def evaluate_v2(model, data, idx, device, amp, batch_size=256, max_items=2000, seed=0):
    model.eval()
    if len(idx) > max_items:
        idx = np.random.default_rng(seed).choice(idx, max_items, replace=False)
    ce, cos, n = 0.0, 0.0, 0
    for i in range(0, len(idx), batch_size):
        batch, _ = data.eval_batch(idx[i:i + batch_size], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        k = len(batch['mz'])
        ce += out['ce'].item() * k
        cos += out['fp_cos'].item() * k
        n += k
    model.train()
    return ce / max(n, 1), cos / max(n, 1)


@torch.no_grad()
def denovo_check_v2(model, data, idx, device, amp, n_samples=20, max_items=200, batch_size=50):
    model.eval()
    if len(idx) > max_items:
        idx = np.random.default_rng(0).choice(idx, max_items, replace=False)
    hit_any, hit_top, valid, total, examples = 0, 0, 0, 0, []
    for i in range(0, len(idx), batch_size):
        batch, s = data.eval_batch(idx[i:i + batch_size], device, with_targets=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            states, mask = model.encode(batch)
            tokens, logprob = model.generate(states, mask, n_samples=n_samples)
        tokens, logprob = tokens.cpu().numpy(), logprob.cpu().numpy()
        for row in range(len(s)):
            order = np.argsort(-logprob[row])
            keys = []
            for j in order:
                if not np.isfinite(logprob[row, j]):
                    continue
                key = plain_key(data.tokenizer.decode(tokens[row, j, 1:]))
                if key is not None:
                    keys.append(key)
            truth = data.ik14[s[row]]
            valid += len(keys) / n_samples
            hit_any += truth in keys
            hit_top += bool(keys) and keys[0] == truth
            total += 1
            if len(examples) < 3:
                examples.append((data.structs.smiles.values[s[row]], data.tokenizer.decode(tokens[row, order[0], 1:])))
    model.train()
    return hit_top / max(total, 1), hit_any / max(total, 1), valid / max(total, 1), examples


# ─────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────

def save_v2_checkpoint(model, opt, cfg, state, name='model_v2.pt', is_best=False):
    os.makedirs(CKPT_DIR, exist_ok=True)
    raw = getattr(model, '_orig_mod', model)
    weights = {'cfg': cfg, 'model': raw.state_dict(), 'state': state}
    path = os.path.join(CKPT_DIR, name)
    torch.save(weights, path)
    if is_best:
        torch.save(weights, os.path.join(CKPT_DIR, 'model_v2_best.pt'))
    # Save last for resume
    torch.save({**weights, 'opt': opt.state_dict()}, os.path.join(CKPT_DIR, 'last_v2.pt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--coco-frac', type=float, default=0.25)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--warmup', type=int, default=1500)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--fp-weight', type=float, default=20.0)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--max-per-struct', type=int, default=6)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.smoke else 'cpu')
    amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)

    log(f'initializing SpectraDataV2 (smoke={args.smoke})...')
    data = SpectraDataV2(max_shards=2 if args.smoke else None, smoke=args.smoke)

    cfg = default_config(len(data.tokenizer))
    cfg['dropout'] = args.dropout
    cfg['vocab'] = data.tokenizer.itos
    if args.smoke:
        cfg.update(d_model=128, n_heads=4, enc_layers=2, dec_layers=2, d_ff=256)
        args.batch_size, args.warmup, args.epochs = 32, 20, 1
        torch.set_num_threads(4)

    model = CasmiModel(cfg).to(device)
    log(f'device {device} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | dropout {args.dropout}')

    decay = [p for n_, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n_, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.98), fused=(device.type == 'cuda')
    )

    steps_per_epoch = len(data.epoch_indices(np.random.default_rng(1), max_per_struct=args.max_per_struct)) // args.batch_size
    total_steps = steps_per_epoch * args.epochs if not args.smoke else 150
    state = {'epoch': 0, 'step': 0, 'history': [], 'best_val_np_ce': float('inf')}

    if args.resume and os.path.exists(os.path.join(CKPT_DIR, 'last_v2.pt')):
        ck = torch.load(os.path.join(CKPT_DIR, 'last_v2.pt'), map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        state = ck['state']
        log(f"resumed V2 training at epoch {state['epoch']} step {state['step']}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    gen = torch.Generator(device=device)
    gen.manual_seed(42)
    t_start, step = time.time(), state['step']
    log(f'steps/epoch {steps_per_epoch:,} | total steps {total_steps:,} | batch {args.batch_size} (COCONUT {args.coco_frac:.0%})')

    model.train()
    for epoch in range(state['epoch'], args.epochs):
        rng = np.random.default_rng(2000 + epoch)
        idx = data.epoch_indices(rng, max_per_struct=args.max_per_struct)
        t_epoch, seen, run = time.time(), 0, {'ce': 0.0, 'bce': 0.0, 'cos': 0.0, 'n': 0}

        for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
            for g in opt.param_groups:
                g['lr'] = lr_at(step)

            batch = data.train_batch(idx[i:i + args.batch_size], device, rng, coco_fraction=args.coco_frac)
            batch = augment_peaks(batch, gen)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(batch)

            loss = out['ce'] + args.fp_weight * out['bce'] + (1.0 - out['fp_cos'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            seen += args.batch_size
            run['ce'] += out['ce'].item()
            run['bce'] += out['bce'].item()
            run['cos'] += out['fp_cos'].item()
            run['n'] += 1

            if step % (10 if args.smoke else 200) == 0:
                elapsed = max(time.time() - t_epoch, 1)
                log(f"ep {epoch} step {step:,}/{total_steps:,} | ce {run['ce'] / run['n']:.4f} | "
                    f"bce {run['bce'] / run['n']:.5f} | fp_cos {run['cos'] / run['n']:.4f} | "
                    f"lr {lr_at(step):.2e} | {seen / elapsed:.0f} items/s")
                run = {'ce': 0.0, 'bce': 0.0, 'cos': 0.0, 'n': 0}

            if args.smoke and step >= total_steps:
                break

        # Epoch evaluation on held-out unseen structures
        np_ce, np_cos = evaluate_v2(model, data, data.val_np_idx, device, amp)
        rd_ce, rd_cos = evaluate_v2(model, data, data.val_rand_idx, device, amp)
        top1, anyhit, valid, examples = denovo_check_v2(
            model, data, data.val_np_idx, device, amp,
            max_items=40 if args.smoke else 200, n_samples=5 if args.smoke else 20
        )

        is_best = np_ce < state['best_val_np_ce']
        if is_best:
            state['best_val_np_ce'] = np_ce

        state.update(epoch=epoch + 1, step=step)
        state['history'].append(dict(
            epoch=epoch + 1, step=step, val_np_ce=np_ce, val_np_fp_cos=np_cos,
            val_rand_ce=rd_ce, val_rand_fp_cos=rd_cos, denovo_top1=top1,
            denovo_any=anyhit, valid_frac=valid, is_best=is_best
        ))

        log(f'== epoch {epoch + 1} done in {(time.time() - t_epoch) / 60:.1f} min | '
            f'val_np ce {np_ce:.4f} fp_cos {np_cos:.4f} {"(NEW BEST)" if is_best else ""} | '
            f'val_rand ce {rd_ce:.4f} fp_cos {rd_cos:.4f} | '
            f'denovo top1 {top1:.3f} any@20 {anyhit:.3f} valid {valid:.2f}')

        for truth, guess in examples:
            log(f'   truth {truth[:70]}\n   guess {guess[:70]}')

        if not args.smoke:
            save_v2_checkpoint(model, opt, cfg, state, name=f'model_v2_ep{epoch + 1}.pt', is_best=is_best)

    log(f'V2 training finished in {(time.time() - t_start) / 3600:.2f} hours. Best val_np CE: {state["best_val_np_ce"]:.4f}')


if __name__ == '__main__':
    main()
