"""
Train the spectrum -> (fingerprint, SMILES) model.

    python -m casmi.train --epochs 12                 # full GPU run
    python -m casmi.train --resume                    # continue from work/ckpt/last.pt
    python -m casmi.train --smoke                     # tiny CPU run that checks the code path end to end

Reads work/structures.parquet + work/spectra/*.npz (see casmi.prep). Writes work/ckpt/{last.pt, model.pt}.
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

from casmi.common import SmilesTokenizer, PAD_ID, FP_BITS, plain_key
from casmi.model import CasmiModel, default_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, 'work')
CKPT_DIR = os.path.join(WORK, 'ckpt')
LIB_E180, LIB_NP = 0, 9            # ids from casmi.prep.LIBS


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────

class SpectraData:
    """All training-eligible spectra as flat numpy arrays + per-structure targets."""

    def __init__(self, max_shards=None):
        structs = pd.read_parquet(os.path.join(WORK, 'structures.parquet'))
        self.structs = structs
        self.tokenizer = SmilesTokenizer.load(os.path.join(WORK, 'vocab.json'))
        n = len(structs)
        max_tok = int(structs.tokens[structs.trainable].map(len).max())
        self.tokens = np.zeros((n, max_tok), np.int64)
        for i, t in enumerate(structs.tokens):
            if t is not None and len(t) <= max_tok:
                self.tokens[i, :len(t)] = t
        self.tok_len = (self.tokens != PAD_ID).sum(1)
        max_bits = int(structs.fp_bits.map(lambda b: len(b) if b is not None else 0).max())
        self.fp_bits = np.full((n, max_bits), -1, np.int64)
        for i, b in enumerate(structs.fp_bits):
            if b is not None:
                self.fp_bits[i, :len(b)] = b
        self.split = structs.split.values
        self.ik14 = structs.ik14.values

        paths = sorted(glob.glob(os.path.join(WORK, 'spectra', 'shard_*.npz')))
        if max_shards:
            paths = paths[-max_shards:]
        parts = [np.load(p) for p in paths]
        cat = lambda k: np.concatenate([p[k] for p in parts])
        self.mz, self.inten, self.prec = cat('mz'), cat('inten'), cat('prec')
        self.adduct, self.ce, self.lib, self.sidx = cat('adduct'), cat('ce'), cat('lib'), cat('sidx')
        spec_split = self.split[self.sidx]
        self.train_idx = np.flatnonzero(spec_split == 'train')
        self.val_np_idx = np.flatnonzero((spec_split == 'val_np') & (self.lib == LIB_NP))
        self.val_rand_idx = np.flatnonzero(spec_split == 'val_rand')
        log(f'spectra {len(self.sidx):,} | train {len(self.train_idx):,} | val_np {len(self.val_np_idx):,} '
            f'| val_rand {len(self.val_rand_idx):,} | vocab {len(self.tokenizer)}')

    def epoch_indices(self, rng, e180_fraction=0.4, np_upsample=8):
        lib = self.lib[self.train_idx]
        other = self.train_idx[lib != LIB_E180]
        e180 = self.train_idx[lib == LIB_E180]
        e180 = rng.choice(e180, size=int(len(e180) * e180_fraction), replace=False)
        np_ex = np.repeat(self.train_idx[lib == LIB_NP], np_upsample)
        idx = np.concatenate([other, e180, np_ex])
        rng.shuffle(idx)
        return idx

    def batch(self, idx, device, with_targets=True):
        idx = np.sort(idx)
        s = self.sidx[idx]
        out = {
            'mz': torch.from_numpy(self.mz[idx]), 'inten': torch.from_numpy(self.inten[idx].astype(np.float32)),
            'prec': torch.from_numpy(self.prec[idx]), 'adduct': torch.from_numpy(self.adduct[idx].astype(np.int64)),
            'ce': torch.from_numpy(self.ce[idx]),
        }
        if with_targets:
            width = int(self.tok_len[s].max())
            out['tokens'] = torch.from_numpy(self.tokens[s, :width])
            out['fp_bits'] = torch.from_numpy(self.fp_bits[s])
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
        if with_targets:
            bits = out.pop('fp_bits')
            fp = torch.zeros(len(idx), FP_BITS + 1, device=device)
            fp.scatter_(1, bits + 1, 1.0)               # padding (-1) lands in column 0, dropped below
            out['fp'] = fp[:, 1:]
        return out, s


def augment(batch, rng_gen):
    """Make clean library spectra look more like noisy timsTOF ones: drop peaks, jitter intensities, add noise peaks."""
    mz, inten, prec = batch['mz'], batch['inten'], batch['prec']
    b, p = mz.shape
    real = inten > 0
    base = inten >= inten.max(dim=1, keepdim=True).values
    drop_p = torch.rand(b, 1, device=mz.device, generator=rng_gen) * 0.3
    keep = (torch.rand(b, p, device=mz.device, generator=rng_gen) > drop_p) | base
    inten = inten * (real & keep)
    jitter = torch.exp(torch.randn(b, p, device=mz.device, generator=rng_gen) * 0.3)
    inten = inten * jitter
    # noise peaks go into empty slots of half of the spectra
    empty = inten <= 0
    noisy_row = torch.rand(b, 1, device=mz.device, generator=rng_gen) < 0.5
    fill_p = torch.rand(b, 1, device=mz.device, generator=rng_gen) * 0.3
    add = empty & noisy_row & (torch.rand(b, p, device=mz.device, generator=rng_gen) < fill_p)
    noise_mz = 50.0 + torch.rand(b, p, device=mz.device, generator=rng_gen) * (prec[:, None].float() - 50.0).clamp(min=1.0)
    noise_int = 0.002 + torch.rand(b, p, device=mz.device, generator=rng_gen) ** 2 * 0.05
    mz = torch.where(add, noise_mz, mz)
    inten = torch.where(add, noise_int, inten)
    inten = inten / inten.max(dim=1, keepdim=True).values.clamp(min=1e-6)
    batch['mz'], batch['inten'] = mz, inten
    return batch


# ─────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, idx, device, amp, batch_size=256, max_items=3000, seed=0):
    model.eval()
    if len(idx) > max_items:
        idx = np.random.default_rng(seed).choice(idx, max_items, replace=False)
    ce, cos, n = 0.0, 0.0, 0
    for i in range(0, len(idx), batch_size):
        batch, _ = data.batch(idx[i:i + batch_size], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(batch)
        k = len(batch['mz'])
        ce, cos, n = ce + out['ce'].item() * k, cos + out['fp_cos'].item() * k, n + k
    model.train()
    return ce / max(n, 1), cos / max(n, 1)


@torch.no_grad()
def denovo_check(model, data, idx, device, amp, n_samples=20, max_items=300, batch_size=50):
    """Sampled exact-match rate on held-out spectra — a cheap health check of generation."""
    model.eval()
    if len(idx) > max_items:
        idx = np.random.default_rng(0).choice(idx, max_items, replace=False)
    hit_any, hit_top, valid, total, examples = 0, 0, 0, 0, []
    for i in range(0, len(idx), batch_size):
        batch, s = data.batch(idx[i:i + batch_size], device, with_targets=False)
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
# Main
# ─────────────────────────────────────────────────────────────────────

def save_checkpoint(model, opt, cfg, state, full=True):
    os.makedirs(CKPT_DIR, exist_ok=True)
    raw = getattr(model, '_orig_mod', model)
    weights = {'cfg': cfg, 'model': raw.state_dict(), 'state': state}
    tmp = os.path.join(CKPT_DIR, 'model.pt.tmp')
    torch.save(weights, tmp)
    os.replace(tmp, os.path.join(CKPT_DIR, 'model.pt'))
    if full:
        tmp = os.path.join(CKPT_DIR, 'last.pt.tmp')
        torch.save({**weights, 'opt': opt.state_dict()}, tmp)
        os.replace(tmp, os.path.join(CKPT_DIR, 'last.pt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--warmup', type=int, default=1500)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--fp-weight', type=float, default=20.0)
    ap.add_argument('--max-hours', type=float, default=1e9)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.smoke else 'cpu')
    amp = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    data = SpectraData(max_shards=2 if args.smoke else None)
    cfg = default_config(len(data.tokenizer))
    if args.smoke:
        cfg.update(d_model=128, n_heads=4, enc_layers=2, dec_layers=2, d_ff=256)
        args.batch_size, args.warmup, args.epochs = 32, 20, 1
        torch.set_num_threads(4)

    model = CasmiModel(cfg).to(device)
    log(f'device {device} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | cfg {json.dumps(cfg)}')
    decay = [p for n_, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n_, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=device.type == 'cuda')

    rng = np.random.default_rng(0)
    steps_per_epoch = len(data.epoch_indices(np.random.default_rng(1))) // args.batch_size
    total_steps = steps_per_epoch * args.epochs if not args.smoke else 150
    state = {'epoch': 0, 'step': 0, 'history': []}
    if args.resume and os.path.exists(os.path.join(CKPT_DIR, 'last.pt')):
        ck = torch.load(os.path.join(CKPT_DIR, 'last.pt'), map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); state = ck['state']
        log(f"resumed at epoch {state['epoch']} step {state['step']}")
    train_model = torch.compile(model, dynamic=True) if args.compile else model

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = min(1.0, (step - args.warmup) / max(1, total_steps - args.warmup))
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))

    gen = torch.Generator(device=device); gen.manual_seed(0)
    t_start, step = time.time(), state['step']
    log(f'steps/epoch {steps_per_epoch:,} | total steps {total_steps:,}')
    model.train()
    for epoch in range(state['epoch'], args.epochs):
        idx = data.epoch_indices(np.random.default_rng(1000 + epoch))
        t_epoch, seen, run = time.time(), 0, {'ce': 0.0, 'bce': 0.0, 'cos': 0.0, 'n': 0}
        for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            batch, _ = data.batch(idx[i:i + args.batch_size], device)
            batch = augment(batch, gen)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = train_model(batch)
            loss = out['ce'] + args.fp_weight * out['bce'] + (1.0 - out['fp_cos'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1; seen += args.batch_size
            run['ce'] += out['ce'].item(); run['bce'] += out['bce'].item(); run['cos'] += out['fp_cos'].item(); run['n'] += 1
            if step % (10 if args.smoke else 200) == 0:
                log(f"ep {epoch} step {step:,}/{total_steps:,} | ce {run['ce'] / run['n']:.4f} | bce {run['bce'] / run['n']:.5f} "
                    f"| fp_cos {run['cos'] / run['n']:.4f} | lr {lr_at(step):.2e} | {seen / (time.time() - t_epoch):.0f} spec/s")
                run = {'ce': 0.0, 'bce': 0.0, 'cos': 0.0, 'n': 0}
            if args.smoke and step >= total_steps:
                break
            if (time.time() - t_start) / 3600 > args.max_hours:
                break

        np_ce, np_cos = evaluate(model, data, data.val_np_idx, device, amp)
        rd_ce, rd_cos = evaluate(model, data, data.val_rand_idx, device, amp)
        top1, anyhit, valid, examples = denovo_check(model, data, data.val_np_idx, device, amp,
                                                     max_items=40 if args.smoke else 300, n_samples=5 if args.smoke else 20)
        state.update(epoch=epoch + 1, step=step)
        state['history'].append(dict(epoch=epoch + 1, step=step, val_np_ce=np_ce, val_np_fp_cos=np_cos, val_rand_ce=rd_ce,
                                     val_rand_fp_cos=rd_cos, denovo_top1=top1, denovo_any=anyhit, valid_frac=valid))
        log(f'== epoch {epoch + 1} done in {(time.time() - t_epoch) / 60:.1f} min | val_np ce {np_ce:.4f} fp_cos {np_cos:.4f} '
            f'| val_rand ce {rd_ce:.4f} fp_cos {rd_cos:.4f} | denovo top1 {top1:.3f} any@20 {anyhit:.3f} valid {valid:.2f}')
        for truth, guess in examples:
            log(f'   truth {truth[:70]}\n            guess {guess[:70]}')
        if not args.smoke:
            save_checkpoint(model, opt, cfg, state)
        if (time.time() - t_start) / 3600 > args.max_hours or (args.smoke and step >= total_steps):
            break

    if args.smoke:
        # KV-cache generation must agree with teacher-forced scoring of the same sequences.
        batch, _ = data.batch(data.val_rand_idx[:4], device, with_targets=False)
        model.eval()
        states, mask = model.encode(batch)
        tokens, logprob = model.generate(states, mask, n_samples=3, max_new_tokens=40)
        flat = tokens.reshape(-1, tokens.shape[-1])
        rescored, _ = model.score(states.repeat_interleave(3, 0), mask.repeat_interleave(3, 0), flat)
        finished = torch.isfinite(logprob.reshape(-1))
        gap = (rescored - logprob.reshape(-1))[finished].abs().max().item() if finished.any() else float('nan')
        log(f'generate-vs-score max |logprob gap| over {int(finished.sum())} finished samples: {gap:.2e} (must be ~0)')
    log('training finished')


if __name__ == '__main__':
    main()
