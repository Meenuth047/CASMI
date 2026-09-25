"""
Enveda CASMI 2026 — Molecule ID From Mass Spectra — V1 Solution
================================================================
Transformer-based de novo spectrum → SMILES generation.

Architecture:
  - PeakEmbedder: sinusoidal m/z encoding + intensity → d_model vectors
  - SpectrumEncoder: 6-layer Transformer encoder over peak embeddings
  - SmilesDecoder: 6-layer Transformer decoder, autoregressive SMILES generation

Usage:
  1. Download the competition data from Kaggle and place train.parquet,
     test.parquet, and sample_submission.csv in the DATA_DIR below.
  2. Run: python casmi_v1.py
  3. Find submission.csv in OUTPUT_DIR.

On Kaggle, set RUNNING_ON_KAGGLE = True (auto-detected).
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing

from rdkit.Chem import MolFromSmiles, MolToSmiles, MolToInchiKey
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl

import lightning as L
from lightning.pytorch import loggers

tqdm.pandas()

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

# Auto-detect environment
RUNNING_ON_KAGGLE = os.path.exists('/kaggle/input')

if RUNNING_ON_KAGGLE:
    COMP_DIR = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'
    OUTPUT_DIR = '/kaggle/working'
else:
    # Local paths — data files are in the same directory as this script
    COMP_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(COMP_DIR, 'train.parquet')
TEST_PATH = os.path.join(COMP_DIR, 'test.parquet')
SAMPLE_SUBMISSION_PATH = os.path.join(COMP_DIR, 'sample_submission.csv')
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, 'submission.csv')

# ── Data processing params ──
MAX_LEN = 128           # max number of peaks per spectrum (truncate to top-intensity)
BPE_PAD_ID = 0
BPE_BOS_ID = 1
BPE_EOS_ID = 2
VOCAB_SIZE = 512

# ── DataLoader params ──
PRECURSOR_INTENSITY = 2.0   # synthetic intensity for prepended precursor peak
MZ_PAD_ID = 0
INTENSITY_PAD_ID = 0
BATCH_SIZE = 128
NUM_WORKERS = min(3, os.cpu_count() or 1)

# ── Model hyperparams ──
EMBED_DIM = 768
DIM_FEEDFORWARD = 3072
DROPOUT = 0.1

ENCODER_N_HEADS = 12
ENCODER_N_LAYERS = 6
ENCODER_ACTIVATION = 'gelu'

DECODER_N_HEADS = 12
DECODER_N_LAYERS = 6
DECODER_ACTIVATION = 'gelu'

N_SAMPLES_VAL = 10
N_SAMPLES_PRED = 25

LEARNING_RATE = 5e-5

# ── Training params ──
MAX_EPOCHS = 1
MAX_TRAIN_SPECTRA = 200_000   # cap training set for speed; set to None to use all
PREDICT_BATCH_SIZE = 32       # smaller batch for generation (expands by n_samples)

# ═══════════════════════════════════════════════════════════════════
# Suppress RDKit warnings (we decode many invalid SMILES during training)
# ═══════════════════════════════════════════════════════════════════
_logger = rkl.logger()
_logger.setLevel(rkl.ERROR)
rkrb.DisableLog("rdApp.error")


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════

def load_and_split_data():
    """Load training data, filter, and split into train/val by structure."""
    print("Loading training data...")
    needed_cols = [
        'ingest_lib', 'inchikey14', 'normalized_smiles', 'precursor_mz',
        'ms2_mzs', 'ms2_normalized_intensities'
    ]
    # Exclude enveda-180: same instrument as test but synthetic drug-like compounds,
    # very different chemistry from natural-product test set. Keeping it adds noise
    # more than signal for this simple single-epoch baseline.
    casmi_all_df = pd.read_parquet(
        TRAIN_PATH, columns=needed_cols,
        filters=[('ingest_lib', '!=', 'enveda-180')]
    )

    # Split by unique structures so val molecules are never seen during training
    all_structs = casmi_all_df.inchikey14.unique()
    train_structs, val_structs = train_test_split(
        all_structs, test_size=500, random_state=0
    )
    train_df = casmi_all_df[casmi_all_df.inchikey14.isin(train_structs)]
    val_df = casmi_all_df[casmi_all_df.inchikey14.isin(val_structs)]

    # Cap training set for faster iteration
    if MAX_TRAIN_SPECTRA is not None and len(train_df) > MAX_TRAIN_SPECTRA:
        train_df = train_df.sample(n=MAX_TRAIN_SPECTRA, random_state=0)

    # One spectrum per val structure (generation is expensive)
    val_df = val_df.drop_duplicates(subset=['inchikey14'], keep='first')

    print(f"  Train: {len(train_df):,} spectra, {train_df.inchikey14.nunique():,} structures")
    print(f"  Val:   {len(val_df):,} spectra, {val_df.inchikey14.nunique():,} structures")
    return train_df, val_df


def process_spectra(df):
    """Sort peaks by descending intensity, truncate to MAX_LEN-1, renormalize."""
    def _process_spectrum(row):
        sort_mask = np.argsort(row.ms2_normalized_intensities)[::-1]
        sorted_mzs = row.ms2_mzs[sort_mask]
        sorted_ints = row.ms2_normalized_intensities[sort_mask]

        # Reserve 1 slot for the prepended precursor
        truncated_mzs = sorted_mzs[:MAX_LEN - 1]
        truncated_ints = sorted_ints[:MAX_LEN - 1]

        # Re-normalize so base peak = 1.0
        normalized_ints = truncated_ints / truncated_ints.max()
        return truncated_mzs, normalized_ints

    mzs, ints = zip(*df.progress_apply(_process_spectrum, axis=1))
    df = df.copy()
    df['processed_mzs'] = list(mzs)
    df['processed_intensities'] = list(ints)
    return df


def train_bpe_tokenizer(structures):
    """Train a BPE tokenizer on the training SMILES strings."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    full_alphabet = list(set(list("".join(structures))))
    tokenizer = Tokenizer(models.BPE())
    special_tokens = [
        ("<pad>", BPE_PAD_ID),
        ("<s>", BPE_BOS_ID),
        ("</s>", BPE_EOS_ID),
    ]
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=special_tokens
    )
    trainer_obj = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        initial_alphabet=full_alphabet,
        special_tokens=["<pad>", "<s>", "</s>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(structures, trainer_obj)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return tokenizer


def tokenize_df_smiles(df, tokenizer):
    """Tokenize the normalized_smiles column with the BPE tokenizer."""
    df = df.copy()
    df['bpe_tokenized_smiles'] = df.normalized_smiles.progress_apply(
        lambda x: np.array(tokenizer.encode(x).ids, dtype=int)
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET & DATAMODULE
# ═══════════════════════════════════════════════════════════════════

class PandasDataset(Dataset):
    def __init__(self, df, column_list, shuffle=True):
        self.df = df[column_list].sample(frac=1, random_state=0) if shuffle else df[column_list]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


class CASMIDataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size, predict_batch_size=None):
        super().__init__()
        self.train_df_raw = train_df
        self.val_df_raw = val_df
        self.test_df_raw = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size or batch_size

    def setup(self, stage):
        train_cols = [
            'precursor_mz', 'processed_mzs', 'processed_intensities',
            'bpe_tokenized_smiles', 'normalized_smiles'
        ]
        if stage == "fit":
            self.train_ds = PandasDataset(self.train_df_raw, column_list=train_cols)
            self.val_ds = PandasDataset(self.val_df_raw, column_list=train_cols)
        elif stage == "validate":
            self.val_ds = PandasDataset(self.val_df_raw, column_list=train_cols)
        elif stage == "predict":
            pred_cols = ['molecule_id', 'precursor_mz', 'processed_mzs', 'processed_intensities']
            self.predict_ds = PandasDataset(self.test_df_raw, column_list=pred_cols, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            collate_fn=self._make_collator('fit'),
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            collate_fn=self._make_collator('validate'),
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, batch_size=self.predict_batch_size,
            collate_fn=self._make_collator('predict'),
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
        )

    def _make_collator(self, stage):
        return partial(self._collator, stage=stage)

    def _collator(self, data, stage='fit'):
        # Prepend precursor m/z as the first peak with a special intensity of 2.0
        mzs = [[row['precursor_mz']] + row['processed_mzs'].tolist() for row in data]
        ints = [[PRECURSOR_INTENSITY] + row['processed_intensities'].tolist() for row in data]

        if stage in ('fit', 'validate'):
            labels = [row['bpe_tokenized_smiles'] for row in data]

        # Pad peaks to max length in batch
        max_peak_len = max(len(x) for x in mzs)
        mzs = [list(x) + [MZ_PAD_ID] * (max_peak_len - len(x)) for x in mzs]
        ints = [list(x) + [INTENSITY_PAD_ID] * (max_peak_len - len(x)) for x in ints]
        mz_array = torch.tensor(mzs)
        intensity_array = torch.tensor(ints)
        attention_mask = torch.where(mz_array == MZ_PAD_ID, 0, 1)

        batch = {
            'mzs': mz_array,
            'intensities': intensity_array,
            'attention_mask': attention_mask,
        }

        if stage in ('fit', 'validate'):
            max_smiles_len = max(len(x) for x in labels)
            labels = [list(x) + [BPE_PAD_ID] * (max_smiles_len - len(x)) for x in labels]
            batch['structure_tokens'] = torch.tensor(labels)
        if stage == 'validate':
            batch['smiles'] = [row['normalized_smiles'] for row in data]
        if stage == 'predict':
            batch['molecule_id'] = [row['molecule_id'] for row in data]

        return batch


# ═══════════════════════════════════════════════════════════════════
# 3. MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════

class PeakEmbedder(nn.Module):
    """Embed (m/z, intensity) peak pairs with sinusoidal m/z encoding (Voronov et al.)."""
    def __init__(self, d_model, dropout, sin_dim=None, mz_log_lims=(-2., 3.), mz_log_power=1.0):
        super().__init__()
        sin_dim = sin_dim if sin_dim is not None else d_model
        self.dropout = dropout

        wavelength = torch.pow(
            10,
            (mz_log_lims[1] - mz_log_lims[0]) * torch.pow(
                torch.linspace(0, 1, int(sin_dim / 2)),
                mz_log_power,
            ) + mz_log_lims[0],
        )
        frequency = 2 * np.pi / wavelength
        self._frequency = nn.Parameter(frequency, requires_grad=False)

        self._ff_block_1 = nn.Sequential(
            nn.Linear(sin_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self._ff_block_2 = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, mz_tensor, intensity_tensor):
        omega_mz = self._frequency.view(
            *(1 for _ in range(mz_tensor.ndim)), -1
        ) * mz_tensor.unsqueeze(-1)
        sin = torch.sin(omega_mz)
        cos = torch.cos(omega_mz)
        mz_vecs = torch.cat([sin, cos], dim=-1)
        mz_embeds = self._ff_block_1(mz_vecs)

        peak_embeds = torch.cat([mz_embeds, intensity_tensor.unsqueeze(2)], dim=2)
        return self._ff_block_2(peak_embeds)


class SpectrumEncoder(nn.Module):
    """Standard Transformer encoder over peak embeddings."""
    def __init__(self, embed_dim, n_heads, n_layers, dim_feedforward=None, dropout=0.1, activation='relu'):
        super().__init__()
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * embed_dim
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim, n_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
                activation=activation,
            ),
            n_layers,
        )
        self._init_weights()

    def forward(self, sequence_input, attention_mask):
        pad_mask = (attention_mask == 0)
        return self.encoder(sequence_input, src_key_padding_mask=pad_mask)

    def _init_weights(self):
        for layer in self.encoder.layers:
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.constant_(mod.weight, 1.0)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)


class SmilesDecoder(nn.Module):
    """Autoregressive Transformer decoder for SMILES generation."""
    def __init__(
        self, embed_dim, vocab_size, n_layers, n_heads,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        dim_feedforward=None, dropout=0.1, activation='gelu',
        validate_n_samples=10, predict_n_samples=25,
    ):
        super().__init__()
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * embed_dim
        self.n_samples_per_stage = {
            'validate': validate_n_samples,
            'predict': predict_n_samples,
        }

        self.wte = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embed_dim,
                dim_feedforward=dim_feedforward,
                nhead=n_heads,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.zeros_(self.lm_head.weight)
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        for layer in self.decoder.layers:
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    nn.init.xavier_uniform_(mod.weight)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)
                elif isinstance(mod, nn.LayerNorm):
                    nn.init.constant_(mod.weight, 1.0)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, 0.0)

    def forward(self, idx, encoder_outputs, encoder_attention_mask, structure_tokens=None):
        encoder_pad_mask = (encoder_attention_mask == 0)
        softcap = 15
        if structure_tokens is not None:
            # Teacher forcing: feed tokens[:-1] and predict tokens[1:]. Without this shift position i is
            # trained to output the token it was just given, which a causal decoder solves by copying.
            idx = idx[:, :-1]
            structure_tokens = structure_tokens[:, 1:]
        tgt = self.wte(idx)

        # Always causal: generation now feeds the whole prefix, so it needs the same mask as training.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1], device=tgt.device)

        x = self.decoder(
            tgt=tgt,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=encoder_pad_mask,
        )
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits / softcap)
        logits = logits.float()

        output_dict = {'logits': logits}
        if structure_tokens is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                structure_tokens.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction='mean',
            )
            output_dict['loss'] = loss
        return output_dict

    @torch.inference_mode()
    def generate(self, encoder_states, encoder_mask, n_samples=1, max_new_tokens=50, temperature=1.0):
        """Sample-based generation with temperature control."""
        B, S, D = encoder_states.shape

        idx = torch.ones_like(encoder_states[:, :1, 0]).long() * self.bos_token_id
        logprob = torch.zeros_like(encoder_states[:, :1, 0])

        idx = idx.expand(-1, n_samples).reshape(B * n_samples, -1)
        idx_next = idx
        logprob_expanded = logprob.expand(-1, n_samples).reshape(B * n_samples, -1)
        eos_generated = idx[:, -1] == self.eos_token_id

        encoder_states = (
            encoder_states.unsqueeze(1)
            .expand(-1, n_samples, -1, -1)
            .reshape(1, B * n_samples, S, D)
            .squeeze(0)
        )
        encoder_mask = (
            encoder_mask.unsqueeze(1)
            .expand(-1, n_samples, -1)
            .reshape(1, B * n_samples, S)
            .squeeze(0)
        )

        for _ in range(max_new_tokens):
            # Feed the full prefix, not just the last token: the decoder has no cache, so it must see
            # everything generated so far to condition on it.
            logits = self.forward(idx, encoder_states, encoder_mask, structure_tokens=None)['logits']
            rescaled_logits = (logits[:, -1, :] / temperature).log_softmax(dim=-1)
            logits_raw = logits[:, -1, :].log_softmax(dim=-1)

            if eos_generated.sum() > 0:
                logits_raw[eos_generated, :] = -float("Inf")
                logits_raw[eos_generated, self.eos_token_id] = 0
                rescaled_logits[eos_generated, :] = -float("Inf")
                rescaled_logits[eos_generated, self.eos_token_id] = 0

            idx_next = torch.multinomial(rescaled_logits.softmax(dim=-1), num_samples=1)
            token_logits_next = torch.take_along_dim(logits_raw, idx_next, dim=1)

            idx = torch.cat((idx, idx_next), dim=1)
            logprob_expanded = torch.cat((logprob_expanded, token_logits_next), dim=1)
            eos_generated = idx[:, -1] == self.eos_token_id

            if eos_generated.sum() == len(eos_generated):
                break
            if (idx == self.eos_token_id).max(axis=-1).values.sum() == idx.shape[0]:
                break

        _, L = idx.shape
        return idx.reshape(B, n_samples, L), logprob_expanded.sum(dim=-1).reshape(B, n_samples)


# ═══════════════════════════════════════════════════════════════════
# 4. LIGHTNING MODULE
# ═══════════════════════════════════════════════════════════════════

# Morgan fingerprints for Tanimoto similarity (validation metric)
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def tanimoto_smiles(smiles1, smiles2):
    mol1, mol2 = MolFromSmiles(smiles1), MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = fp_gen.GetSparseFingerprint(mol1)
    fp2 = fp_gen.GetSparseFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# Will be set after tokenizer is trained
_global_tokenizer = None


def decode_tokenized_smiles(bpe_smiles_tokens):
    return _global_tokenizer.decode(bpe_smiles_tokens).replace(' ', '')


class DeNovoLightningModel(L.LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.peak_embedder = PeakEmbedder(**kwargs['peak_embedder'])
        self.spectrum_encoder = SpectrumEncoder(**kwargs['spectrum_encoder'])
        self.smiles_decoder = SmilesDecoder(**kwargs['smiles_decoder'])
        self.optimizer_config = kwargs['optimizer']

    def forward(self, batch, stage, return_encoder_states=False):
        encoded_spectra = self.spectrum_encoder(
            self.peak_embedder(batch['mzs'], batch['intensities']),
            batch['attention_mask']
        )
        outputs = {}
        if stage in ('fit', 'validate'):
            outputs = self.smiles_decoder(
                batch['structure_tokens'],
                encoded_spectra,
                batch['attention_mask'],
                structure_tokens=batch['structure_tokens'],
            )
        if stage in ('validate', 'predict'):
            tokens, scores = self.smiles_decoder.generate(
                encoded_spectra,
                batch['attention_mask'],
                n_samples=self.smiles_decoder.n_samples_per_stage[stage],
                temperature=1.0,
            )
            outputs['generated_tokens'] = tokens
            outputs['generated_scores'] = scores
        if return_encoder_states:
            outputs['encoder_states'] = encoded_spectra
        return outputs

    def training_step(self, batch):
        outputs = self(batch, "fit")
        loss = outputs['loss']
        self.log(
            "train_loss", loss,
            sync_dist=True, prog_bar=True,
            on_step=True, on_epoch=True,
            logger=True, batch_size=batch['mzs'].shape[0],
        )
        return loss

    def validation_step(self, batch):
        forward_outputs = self(batch, "validate", return_encoder_states=True)
        loss = forward_outputs['loss']
        generated_tokens, generated_scores = self.smiles_decoder.generate(
            forward_outputs['encoder_states'],
            batch['attention_mask'],
            n_samples=10,
            temperature=1.0,
        )
        generation_metrics = self._compute_val_metrics(
            generated_tokens, generated_scores, batch['smiles']
        )
        metrics_dict = {
            "val_loss": loss,
            **{f"val_{k}": v for k, v in generation_metrics.items()},
        }
        self.log_dict(
            metrics_dict,
            add_dataloader_idx=False,
            on_epoch=True, sync_dist=True,
            batch_size=batch['mzs'].shape[0],
        )

    def predict_step(self, batch):
        outputs = self(batch, "predict")
        return batch['molecule_id'], outputs['generated_tokens'], outputs['generated_scores']

    def _compute_val_metrics(self, generated_tokens, generated_scores, labels):
        decoded_smiles = [
            [decode_tokenized_smiles(tokens.tolist()) for tokens in token_seqs]
            for token_seqs in generated_tokens
        ]
        score_masks = [scores.argsort(descending=True) for scores in generated_scores]
        sorted_valid_smiles = [
            [
                smiles_seq[idx]
                for idx in mask
                if smiles_seq[idx] != '' and MolFromSmiles(smiles_seq[idx]) is not None
            ]
            for smiles_seq, mask in zip(decoded_smiles, score_masks)
        ]
        top_valid_smiles = [
            smiles_seq[0] if len(smiles_seq) > 0 else None
            for smiles_seq in sorted_valid_smiles
        ]

        def _sm_to_ikey(smiles):
            return MolToInchiKey(MolFromSmiles(smiles)).split('-')[0]

        ikey_match = [
            _sm_to_ikey(smiles) == _sm_to_ikey(label)
            if smiles is not None else False
            for smiles, label in zip(top_valid_smiles, labels)
        ]
        tanimotos = [
            tanimoto_smiles(smiles, label) if smiles is not None else 0.0
            for smiles, label in zip(top_valid_smiles, labels)
        ]

        return {
            'valid_smiles': np.mean([x is not None for x in top_valid_smiles]),
            'tanimoto': np.mean(tanimotos),
            'exact match': np.mean(ikey_match),
        }

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.optimizer_config['lr'])


# ═══════════════════════════════════════════════════════════════════
# 5. BUILD SUBMISSION
# ═══════════════════════════════════════════════════════════════════

def build_submission(predictions, sample_submission_path, output_path):
    """Aggregate per-spectrum predictions into per-molecule ranked SMILES lists."""
    N_GUESSES = 25
    FALLBACK_SMILES = 'CCO'

    best_by_molecule = defaultdict(dict)  # molecule_id -> {inchikey14: (score, smiles)}
    for molecule_ids, token_batch, score_batch in predictions:
        for molecule_id, token_seqs, scores in zip(molecule_ids, token_batch, score_batch):
            for tokens, score in zip(token_seqs, scores):
                smiles = decode_tokenized_smiles(tokens.tolist())
                mol = MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    continue
                inchikey14 = MolToInchiKey(mol).split('-')[0]
                candidate = (float(score), MolToSmiles(mol))
                if candidate > best_by_molecule[molecule_id].get(inchikey14, (-np.inf, '')):
                    best_by_molecule[molecule_id][inchikey14] = candidate

    def top_guesses(molecule_id):
        ranked = sorted(best_by_molecule.get(molecule_id, {}).values(), reverse=True)
        guesses = [smiles for _, smiles in ranked[:N_GUESSES]]
        return ';'.join(guesses) if guesses else FALLBACK_SMILES

    submission = pd.read_csv(sample_submission_path)
    submission['smiles'] = submission.molecule_id.apply(top_guesses)
    submission.to_csv(output_path, index=False)

    n_guesses = submission.smiles.str.split(';').map(len)
    print(f"\n✅ Wrote {len(submission)} rows to {output_path}")
    print(f"   Guesses per molecule: median {int(n_guesses.median())}, max {n_guesses.max()}")
    return submission


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    global _global_tokenizer

    # Verify data exists
    for path in [TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            print(f"\n   Please download the competition data from Kaggle and place")
            print(f"   train.parquet, test.parquet, and sample_submission.csv in:")
            print(f"   {COMP_DIR}")
            sys.exit(1)

    # ── Load & split ──
    train_df, val_df = load_and_split_data()

    # ── Process spectra ──
    print("\nProcessing spectra...")
    train_df = process_spectra(train_df)
    val_df = process_spectra(val_df)

    # ── Train BPE tokenizer on SMILES ──
    print("\nTraining BPE tokenizer...")
    bpe_train_structures = list(train_df.normalized_smiles.unique())
    tokenizer = train_bpe_tokenizer(bpe_train_structures)
    _global_tokenizer = tokenizer

    # ── Tokenize SMILES ──
    print("\nTokenizing SMILES...")
    train_df = tokenize_df_smiles(train_df, tokenizer)
    val_df = tokenize_df_smiles(val_df, tokenizer)

    # ── Load test set ──
    print("\nLoading test data...")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  Test: {len(test_df)} spectra, {test_df.molecule_id.nunique()} molecules")
    test_df = process_spectra(test_df)

    # ── Build data module ──
    datamodule = CASMIDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        predict_batch_size=PREDICT_BATCH_SIZE,
    )

    # ── Build model ──
    model_params = {
        'peak_embedder': {
            'd_model': EMBED_DIM,
            'dropout': DROPOUT,
        },
        'spectrum_encoder': {
            'embed_dim': EMBED_DIM,
            'n_heads': ENCODER_N_HEADS,
            'n_layers': ENCODER_N_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT,
            'activation': ENCODER_ACTIVATION,
        },
        'smiles_decoder': {
            'embed_dim': EMBED_DIM,
            'vocab_size': VOCAB_SIZE,
            'n_layers': DECODER_N_LAYERS,
            'n_heads': DECODER_N_HEADS,
            'pad_token_id': BPE_PAD_ID,
            'bos_token_id': BPE_BOS_ID,
            'eos_token_id': BPE_EOS_ID,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT,
            'activation': DECODER_ACTIVATION,
            'validate_n_samples': N_SAMPLES_VAL,
            'predict_n_samples': N_SAMPLES_PRED,
        },
        'optimizer': {
            'lr': LEARNING_RATE,
        },
    }
    model = DeNovoLightningModel(model_params)

    # ── Configure trainer ──
    precision = 'bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        '16-mixed' if torch.cuda.is_available() else '32-true'
    )

    logger_path = os.path.join(OUTPUT_DIR, 'lightning_logs')
    tb_logger = loggers.TensorBoardLogger(save_dir=logger_path, name='casmi_v1')

    checkpoint_dir = os.path.join(OUTPUT_DIR, 'model_checkpoints', 'casmi_v1')
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val_tanimoto',
        mode='max',
        every_n_epochs=1,
        save_top_k=5,
        save_last=True,
        filename="{epoch}-{step}-{val_tanimoto:.3f}",
    )

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,
        accelerator='auto',
        precision=precision,
        log_every_n_steps=100,
        check_val_every_n_epoch=1,
        accumulate_grad_batches=3,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=2,
    )

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"Training for {MAX_EPOCHS} epoch(s)...")
    print(f"  Precision: {precision}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Grad accumulation: 3")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Model params: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"{'='*60}\n")

    datamodule.setup('fit')
    trainer.fit(model, datamodule=datamodule)

    # ── Predict ──
    print(f"\n{'='*60}")
    print("Generating predictions on test set...")
    print(f"{'='*60}\n")

    datamodule.setup('predict')
    predict_trainer = L.Trainer(
        accelerator='auto', devices=1, precision=precision, logger=False,
    )
    predictions = predict_trainer.predict(model, dataloaders=datamodule.predict_dataloader())

    # ── Build submission ──
    submission = build_submission(predictions, SAMPLE_SUBMISSION_PATH, SUBMISSION_PATH)
    print("\n🎉 Done! First 5 rows:")
    print(submission.head().to_string())


if __name__ == '__main__':
    main()
