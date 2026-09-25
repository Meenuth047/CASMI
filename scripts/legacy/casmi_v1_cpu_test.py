"""
CASMI 2026 — CPU Test Run (Memory-Efficient)
==============================================
Reads only a SINGLE row group from train.parquet to stay under RAM limits.
Uses a small model (2-layer, 256-dim, ~4M params) on 5K spectra.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing

from rdkit.Chem import MolFromSmiles, MolToSmiles, MolToInchiKey
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

import lightning as L

tqdm.pandas()

# ═══════════════════════════════════════════════════════════════════
# Configuration — REDUCED FOR CPU + LOW RAM
# ═══════════════════════════════════════════════════════════════════

COMP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(COMP_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(COMP_DIR, 'train.parquet')
TEST_PATH = os.path.join(COMP_DIR, 'test.parquet')
SAMPLE_SUBMISSION_PATH = os.path.join(COMP_DIR, 'sample_submission.csv')
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, 'submission.csv')

# Data
MAX_LEN = 64
BPE_PAD_ID = 0
BPE_BOS_ID = 1
BPE_EOS_ID = 2
VOCAB_SIZE = 256
TRAIN_SPECTRA = 5_000
VAL_SPECTRA = 50

# DataLoader
PRECURSOR_INTENSITY = 2.0
MZ_PAD_ID = 0
INTENSITY_PAD_ID = 0
BATCH_SIZE = 32
NUM_WORKERS = 0

# Model — small for CPU
EMBED_DIM = 256
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
ENCODER_N_HEADS = 4
ENCODER_N_LAYERS = 2
ENCODER_ACTIVATION = 'gelu'
DECODER_N_HEADS = 4
DECODER_N_LAYERS = 2
DECODER_ACTIVATION = 'gelu'
N_SAMPLES_VAL = 3
N_SAMPLES_PRED = 5
LEARNING_RATE = 1e-4
MAX_EPOCHS = 1

# Suppress RDKit warnings
_rl = rkl.logger()
_rl.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")

# ═══════════════════════════════════════════════════════════════════
# Memory-efficient data loading: read only 1 row group
# ═══════════════════════════════════════════════════════════════════

def load_train_data_low_memory():
    """Read a single row group from parquet to avoid OOM."""
    import pyarrow.parquet as pq

    print("Loading training data (1 row group only)...")
    needed = ['inchikey14', 'normalized_smiles', 'precursor_mz',
              'ms2_mzs', 'ms2_normalized_intensities', 'ingest_lib']

    pf = pq.ParquetFile(TRAIN_PATH)
    # Row groups 9-20 contain non-enveda-180 data. RG 10 has gnps+masaryk+massbank.
    rg_idx = 10
    table = pf.read_row_group(rg_idx, columns=needed)
    df = table.to_pandas()
    del table
    gc.collect()

    # Filter out enveda-180
    df = df[df.ingest_lib != 'enveda-180'].copy()
    df.drop(columns=['ingest_lib'], inplace=True)

    print(f"  Row group {rg_idx}: {len(df):,} rows after filtering")
    if len(df) == 0:
        print("  ❌ No data after filtering! Trying row group 11...")
        table = pf.read_row_group(11, columns=needed)
        df = table.to_pandas()
        del table
        df = df[df.ingest_lib != 'enveda-180'].copy()
        df.drop(columns=['ingest_lib'], inplace=True)
        print(f"  Row group 11: {len(df):,} rows")

    # Split: take last VAL_SPECTRA unique structures as val
    unique_structs = df.inchikey14.unique()
    np.random.seed(42)
    np.random.shuffle(unique_structs)
    val_structs = set(unique_structs[:VAL_SPECTRA])

    val_df = df[df.inchikey14.isin(val_structs)].drop_duplicates(
        subset=['inchikey14'], keep='first').head(VAL_SPECTRA)
    train_df = df[~df.inchikey14.isin(val_structs)]

    if len(train_df) > TRAIN_SPECTRA:
        train_df = train_df.sample(n=TRAIN_SPECTRA, random_state=0)

    del df
    gc.collect()

    print(f"  Train: {len(train_df):,} spectra, {train_df.inchikey14.nunique():,} structures")
    print(f"  Val:   {len(val_df):,} spectra")
    return train_df, val_df


def process_spectra(df):
    def _process(row):
        sort_mask = np.argsort(row.ms2_normalized_intensities)[::-1]
        mzs = row.ms2_mzs[sort_mask][:MAX_LEN - 1]
        ints = row.ms2_normalized_intensities[sort_mask][:MAX_LEN - 1]
        ints = ints / ints.max()
        return mzs, ints

    mzs, ints = zip(*df.progress_apply(_process, axis=1))
    df = df.copy()
    df['processed_mzs'] = list(mzs)
    df['processed_intensities'] = list(ints)
    # Drop original heavy columns to save memory
    df.drop(columns=['ms2_mzs', 'ms2_normalized_intensities'], inplace=True, errors='ignore')
    return df


def train_bpe_tokenizer(structures):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    full_alphabet = list(set("".join(structures)))
    tokenizer = Tokenizer(models.BPE())
    special_tokens = [("<pad>", BPE_PAD_ID), ("<s>", BPE_BOS_ID), ("</s>", BPE_EOS_ID)]
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=special_tokens)
    trainer_obj = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, initial_alphabet=full_alphabet,
        special_tokens=["<pad>", "<s>", "</s>"], show_progress=True)
    tokenizer.train_from_iterator(structures, trainer_obj)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return tokenizer


# ═══════════════════════════════════════════════════════════════════
# Dataset & DataModule
# ═══════════════════════════════════════════════════════════════════

class SimpleDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(data, stage='fit'):
    mzs = [[row['precursor_mz']] + row['processed_mzs'].tolist() for row in data]
    ints = [[PRECURSOR_INTENSITY] + row['processed_intensities'].tolist() for row in data]

    max_peak_len = max(len(x) for x in mzs)
    mzs = [list(x) + [MZ_PAD_ID] * (max_peak_len - len(x)) for x in mzs]
    ints = [list(x) + [INTENSITY_PAD_ID] * (max_peak_len - len(x)) for x in ints]
    mz_t = torch.tensor(mzs)
    int_t = torch.tensor(ints)
    mask = torch.where(mz_t == MZ_PAD_ID, 0, 1)

    batch = {'mzs': mz_t, 'intensities': int_t, 'attention_mask': mask}

    if stage in ('fit', 'validate'):
        labels = [row['bpe_tokens'] for row in data]
        max_len = max(len(x) for x in labels)
        labels = [list(x) + [BPE_PAD_ID] * (max_len - len(x)) for x in labels]
        batch['structure_tokens'] = torch.tensor(labels)
    if stage == 'validate':
        batch['smiles'] = [row['normalized_smiles'] for row in data]
    if stage == 'predict':
        batch['molecule_id'] = [row['molecule_id'] for row in data]
    return batch


# ═══════════════════════════════════════════════════════════════════
# Model Components
# ═══════════════════════════════════════════════════════════════════

class PeakEmbedder(nn.Module):
    def __init__(self, d_model, dropout, sin_dim=None, mz_log_lims=(-2., 3.), mz_log_power=1.0):
        super().__init__()
        sin_dim = sin_dim or d_model
        wavelength = torch.pow(10, (mz_log_lims[1] - mz_log_lims[0]) * torch.pow(
            torch.linspace(0, 1, sin_dim // 2), mz_log_power) + mz_log_lims[0])
        self._frequency = nn.Parameter(2 * np.pi / wavelength, requires_grad=False)
        self._ff1 = nn.Sequential(
            nn.Linear(sin_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self._ff2 = nn.Sequential(
            nn.Linear(d_model + 1, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, mz, intensity):
        omega = self._frequency.view(*(1 for _ in range(mz.ndim)), -1) * mz.unsqueeze(-1)
        mz_emb = self._ff1(torch.cat([torch.sin(omega), torch.cos(omega)], dim=-1))
        return self._ff2(torch.cat([mz_emb, intensity.unsqueeze(2)], dim=2))


class SpecEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, dim_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        self.enc = TransformerEncoder(
            TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=dim_ff or 4*embed_dim,
                                   batch_first=True, dropout=dropout, activation=activation),
            n_layers)
    def forward(self, x, mask):
        return self.enc(x, src_key_padding_mask=(mask == 0))


class SMIDecoder(nn.Module):
    def __init__(self, embed_dim, vocab_size, n_layers, n_heads,
                 pad_id=0, bos_id=1, eos_id=2, dim_ff=None, dropout=0.1,
                 activation='gelu', val_samples=3, pred_samples=5):
        super().__init__()
        self.bos_id, self.pad_id, self.eos_id = bos_id, pad_id, eos_id
        self.n_samples = {'validate': val_samples, 'predict': pred_samples}
        self.wte = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_dim, dim_feedforward=dim_ff or 4*embed_dim,
                                      nhead=n_heads, dropout=dropout, activation=activation,
                                      batch_first=True),
            num_layers=n_layers)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.zeros_(self.head.weight)
        nn.init.normal_(self.wte.weight, mean=0., std=1.)

    def forward(self, idx, enc_out, enc_mask, structure_tokens=None):
        if structure_tokens is not None:
            # Teacher forcing: feed tokens[:-1], predict tokens[1:] (unshifted labels teach the decoder to copy).
            idx = idx[:, :-1]
            structure_tokens = structure_tokens[:, 1:]
        tgt = self.wte(idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)  # always causal
        x = self.dec(tgt=tgt, memory=enc_out, tgt_mask=tgt_mask,
                     memory_key_padding_mask=(enc_mask == 0))
        logits = (15 * torch.tanh(self.head(x) / 15)).float()
        out = {'logits': logits}
        if structure_tokens is not None:
            out['loss'] = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                          structure_tokens.reshape(-1),
                                          ignore_index=self.pad_id, reduction='mean')
        return out

    @torch.inference_mode()
    def generate(self, enc_states, enc_mask, n_samples=1, max_tokens=50, temperature=1.0):
        B, S, D = enc_states.shape
        idx = (torch.ones(B, 1, dtype=torch.long, device=enc_states.device) * self.bos_id)
        idx = idx.repeat_interleave(n_samples, dim=0)
        logprob = torch.zeros(B * n_samples, 1, device=enc_states.device)
        enc_states = enc_states.repeat_interleave(n_samples, dim=0)
        enc_mask = enc_mask.repeat_interleave(n_samples, dim=0)
        eos_done = torch.zeros(B * n_samples, dtype=torch.bool, device=enc_states.device)

        for _ in range(max_tokens):
            logits = self.forward(idx, enc_states, enc_mask)['logits'][:, -1, :]
            scaled = (logits / temperature).log_softmax(dim=-1)
            raw = logits.log_softmax(dim=-1)
            if eos_done.any():
                scaled[eos_done] = -float('inf')
                scaled[eos_done, self.eos_id] = 0
                raw[eos_done] = -float('inf')
                raw[eos_done, self.eos_id] = 0
            next_tok = torch.multinomial(scaled.softmax(dim=-1), 1)
            tok_lp = torch.gather(raw, 1, next_tok)
            idx = torch.cat([idx, next_tok], dim=1)
            logprob = torch.cat([logprob, tok_lp], dim=1)
            eos_done = (idx[:, -1] == self.eos_id)
            if eos_done.all():
                break
        L = idx.shape[1]
        return idx.reshape(B, n_samples, L), logprob.sum(dim=-1).reshape(B, n_samples)


# ═══════════════════════════════════════════════════════════════════
# Lightning Model
# ═══════════════════════════════════════════════════════════════════

_tokenizer = None

def decode_tokens(tok_list):
    return _tokenizer.decode(tok_list).replace(' ', '')


class CASMIModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.embedder = PeakEmbedder(**cfg['embedder'])
        self.encoder = SpecEncoder(**cfg['encoder'])
        self.decoder = SMIDecoder(**cfg['decoder'])
        self.lr = cfg['lr']

    def forward(self, batch, stage):
        enc = self.encoder(self.embedder(batch['mzs'], batch['intensities']), batch['attention_mask'])
        out = {}
        if stage in ('fit', 'validate'):
            out = self.decoder(batch['structure_tokens'], enc, batch['attention_mask'],
                               structure_tokens=batch['structure_tokens'])
        if stage in ('validate', 'predict'):
            toks, scores = self.decoder.generate(enc, batch['attention_mask'],
                                                  n_samples=self.decoder.n_samples[stage])
            out['gen_tokens'] = toks
            out['gen_scores'] = scores
        return out

    def training_step(self, batch):
        loss = self(batch, 'fit')['loss']
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=batch['mzs'].shape[0])
        return loss

    def validation_step(self, batch):
        out = self(batch, 'validate')
        self.log('val_loss', out['loss'], on_epoch=True, batch_size=batch['mzs'].shape[0])

    def predict_step(self, batch):
        out = self(batch, 'predict')
        return batch['molecule_id'], out['gen_tokens'], out['gen_scores']

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    global _tokenizer

    print("=" * 60)
    print("CASMI 2026 — CPU TEST (Memory-Efficient)")
    print("  Model: 2-layer, 256-dim (~4M params)")
    print("  Data:  5K train from 1 row group, full test")
    print("=" * 60)

    for p in [TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH]:
        if not os.path.exists(p):
            print(f"❌ Missing: {p}"); sys.exit(1)

    # ── Load train (memory efficient) ──
    train_df, val_df = load_train_data_low_memory()
    gc.collect()

    # ── Process spectra ──
    print("\nProcessing spectra...")
    train_df = process_spectra(train_df)
    val_df = process_spectra(val_df)
    gc.collect()

    # ── BPE tokenizer ──
    print("\nTraining BPE tokenizer...")
    tokenizer = train_bpe_tokenizer(list(train_df.normalized_smiles.unique()))
    _tokenizer = tokenizer

    print("Tokenizing SMILES...")
    train_df['bpe_tokens'] = train_df.normalized_smiles.progress_apply(
        lambda x: np.array(tokenizer.encode(x).ids, dtype=np.int32))
    val_df['bpe_tokens'] = val_df.normalized_smiles.progress_apply(
        lambda x: np.array(tokenizer.encode(x).ids, dtype=np.int32))

    # ── Convert to record lists (more memory-efficient than keeping DataFrame) ──
    train_records = train_df[['precursor_mz', 'processed_mzs', 'processed_intensities',
                              'bpe_tokens', 'normalized_smiles']].to_dict('records')
    val_records = val_df[['precursor_mz', 'processed_mzs', 'processed_intensities',
                          'bpe_tokens', 'normalized_smiles']].to_dict('records')
    del train_df, val_df
    gc.collect()

    train_ds = SimpleDataset(train_records)
    val_ds = SimpleDataset(val_records)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          collate_fn=partial(collate_fn, stage='fit'), shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        collate_fn=partial(collate_fn, stage='validate'))

    # ── Build model ──
    cfg = {
        'embedder': {'d_model': EMBED_DIM, 'dropout': DROPOUT},
        'encoder': {'embed_dim': EMBED_DIM, 'n_heads': ENCODER_N_HEADS,
                    'n_layers': ENCODER_N_LAYERS, 'dim_ff': DIM_FEEDFORWARD,
                    'dropout': DROPOUT, 'activation': ENCODER_ACTIVATION},
        'decoder': {'embed_dim': EMBED_DIM, 'vocab_size': VOCAB_SIZE,
                    'n_layers': DECODER_N_LAYERS, 'n_heads': DECODER_N_HEADS,
                    'pad_id': BPE_PAD_ID, 'bos_id': BPE_BOS_ID, 'eos_id': BPE_EOS_ID,
                    'dim_ff': DIM_FEEDFORWARD, 'dropout': DROPOUT,
                    'activation': DECODER_ACTIVATION,
                    'val_samples': N_SAMPLES_VAL, 'pred_samples': N_SAMPLES_PRED},
        'lr': LEARNING_RATE,
    }
    model = CASMIModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params / 1e6:.1f}M parameters")

    # ── Train ──
    print(f"\n{'='*60}")
    print("Training (CPU, 1 epoch)...")
    print(f"{'='*60}\n")

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS, devices=1, accelerator='cpu', precision='32-true',
        log_every_n_steps=10, check_val_every_n_epoch=1,
        accumulate_grad_batches=1, logger=False,
        enable_checkpointing=False, num_sanity_val_steps=1)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    del train_ds, val_ds, train_records, val_records, train_dl, val_dl
    gc.collect()

    # ── Load and process test data ──
    print(f"\n{'='*60}")
    print("Loading test data & generating predictions...")
    print(f"{'='*60}\n")

    test_df = pd.read_parquet(TEST_PATH)
    print(f"  Test: {len(test_df)} spectra, {test_df.molecule_id.nunique()} molecules")
    test_df = process_spectra(test_df)

    test_records = test_df[['molecule_id', 'precursor_mz', 'processed_mzs',
                            'processed_intensities']].to_dict('records')
    del test_df
    gc.collect()

    test_ds = SimpleDataset(test_records)
    test_dl = DataLoader(test_ds, batch_size=16,
                         collate_fn=partial(collate_fn, stage='predict'))

    pred_trainer = L.Trainer(accelerator='cpu', devices=1, precision='32-true', logger=False)
    predictions = pred_trainer.predict(model, dataloaders=test_dl)

    # ── Build submission ──
    print("\nBuilding submission...")
    best_by_mol = defaultdict(dict)
    for mol_ids, tok_batch, score_batch in predictions:
        for mol_id, tok_seqs, scores in zip(mol_ids, tok_batch, score_batch):
            for tokens, score in zip(tok_seqs, scores):
                smiles = decode_tokens(tokens.tolist())
                mol = MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    continue
                ik14 = MolToInchiKey(mol).split('-')[0]
                candidate = (float(score), MolToSmiles(mol))
                if candidate > best_by_mol[mol_id].get(ik14, (-np.inf, '')):
                    best_by_mol[mol_id][ik14] = candidate

    def top_guesses(mol_id):
        ranked = sorted(best_by_mol.get(mol_id, {}).values(), reverse=True)
        guesses = [s for _, s in ranked[:25]]
        return ';'.join(guesses) if guesses else 'CCO'

    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission['smiles'] = submission.molecule_id.apply(top_guesses)
    submission.to_csv(SUBMISSION_PATH, index=False)

    n_guesses = submission.smiles.str.split(';').map(len)
    print(f"\n✅ Wrote {len(submission)} rows to {SUBMISSION_PATH}")
    print(f"   Guesses per molecule: median {int(n_guesses.median())}, max {n_guesses.max()}")
    print("\nFirst 5 rows:")
    print(submission.head().to_string())
    print("\n⚠️  This was a TEST RUN. For real submission, use casmi_v1_kaggle.ipynb on Kaggle GPU.")


if __name__ == '__main__':
    main()
