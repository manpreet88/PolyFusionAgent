# fingerprint_mlm_training.py
import os
import json
import time
import shutil
import sys
import csv

# Increase max CSV field size limit (some fingerprint fields can be long)
csv.field_size_limit(sys.maxsize)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score
from typing import List

# ---------------------------
# Configuration / Constants
# ---------------------------
# MLM mask probability
P_MASK = 0.15

# Fingerprint specifics
FINGERPRINT_KEY = "morgan_r3_bits"   # inside the JSON stored under 'fingerprints' column
FP_LENGTH = 2048                      # expected fingerprint vector length (bits)
# Vocabulary: {0, 1, MASK} where 0/1 are real bits and MASK token id = 2 used as masked input
MASK_TOKEN_ID = 2
VOCAB_SIZE = 3

# Model / encoder hyperparams
HIDDEN_DIM = 256
TRANSFORMER_NUM_LAYERS = 4
TRANSFORMER_NHEAD = 8
TRANSFORMER_FF = 1024
DROPOUT = 0.1

# Training / data hyperparams
TRAIN_BATCH_SIZE = 16   # number of molecules per batch
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# File locations (changed as requested)
CSV_PATH = "Polymer_Foundational_Model/polymer_structures_unified_processed.csv"
OUTPUT_DIR = "./fingerprint_mlm_output_5M"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 1. Load Data (chunked to avoid OOM) - read fingerprints column
# ---------------------------
TARGET_ROWS = 5000000
CHUNKSIZE = 50000

fp_lists: List[List[int]] = []
rows_read = 0

# Expect 'fingerprints' column value to be a JSON string we can json.loads()
# that contains e.g. {"morgan_r3_bits": ["0","1","0",...]}
for chunk in pd.read_csv(CSV_PATH, engine="python", chunksize=CHUNKSIZE):
    # some files might already have parsed JSON-like dicts; ensure we handle strings
    fps_chunk = chunk["fingerprints"]
    for fpval in fps_chunk:
        if pd.isna(fpval):
            # skip or use zeros
            fp_lists.append([0] * FP_LENGTH)
            continue

        # If it's already a dict-like object, use directly; else parse JSON string
        if isinstance(fpval, str):
            try:
                fp_json = json.loads(fpval)
            except Exception:
                # fallback: try to fix common quoting issues
                try:
                    fp_json = json.loads(fpval.replace("'", '"'))
                except Exception:
                    # as last fallback, treat the string as a comma separated "0,1,0,..."
                    parts = [p.strip().strip('"').strip("'") for p in fpval.split(",")]
                    bits = [1 if p in ("1", "True", "true") else 0 for p in parts[:FP_LENGTH]]
                    if len(bits) < FP_LENGTH:
                        bits += [0] * (FP_LENGTH - len(bits))
                    fp_lists.append(bits)
                    continue
        elif isinstance(fpval, dict):
            fp_json = fpval
        else:
            # Unknown type, zero pad
            fp_lists.append([0] * FP_LENGTH)
            continue

        # Extract the fingerprint bit list
        bits = fp_json.get(FINGERPRINT_KEY, None)
        if bits is None:
            # fallback if top-level is already list
            if isinstance(fp_json, list):
                bits = fp_json
            else:
                # default zero vector
                bits = [0] * FP_LENGTH

        # bits may be list of strings "0"/"1" or ints
        # normalize to ints and ensure length
        normalized = []
        for b in bits:
            if isinstance(b, str):
                b_clean = b.strip().strip('"').strip("'")
                normalized.append(1 if b_clean in ("1", "True", "true") else 0)
            elif isinstance(b, (int, np.integer)):
                normalized.append(1 if int(b) != 0 else 0)
            else:
                normalized.append(0)
            if len(normalized) >= FP_LENGTH:
                break

        if len(normalized) < FP_LENGTH:
            # pad with zeros
            normalized.extend([0] * (FP_LENGTH - len(normalized)))

        fp_lists.append(normalized[:FP_LENGTH])

    rows_read += len(chunk)
    if rows_read >= TARGET_ROWS:
        break

print(f"Loaded {len(fp_lists)} fingerprint vectors (using FP_LENGTH={FP_LENGTH}).")

# ---------------------------
# 2. Train/Val Split
# ---------------------------
train_idx, val_idx = train_test_split(list(range(len(fp_lists))), test_size=0.2, random_state=42)
train_fps = [torch.tensor(fp_lists[i], dtype=torch.long) for i in train_idx]
val_fps   = [torch.tensor(fp_lists[i], dtype=torch.long) for i in val_idx]

# ---------------------------
# Compute class weights (for weighted CE to mitigate bit imbalance)
# (we compute but will not apply them to match previous MLM-style loss behavior)
# ---------------------------
# We'll compute weights for classes {0,1} only (targets).
counts = np.ones((2,), dtype=np.float64)  # initialize with 1 to avoid zero division
for fp in train_fps:
    vals = fp.cpu().numpy().astype(int)
    counts[0] += np.sum(vals == 0)
    counts[1] += np.sum(vals == 1)

freq = counts / counts.sum()
inv_freq = 1.0 / (freq + 1e-12)
class_weights_arr = inv_freq / inv_freq.mean()
class_weights = torch.tensor(class_weights_arr, dtype=torch.float)  # shape [2]
print("Class weights (for bit 0 and bit 1):", class_weights.numpy())

# ---------------------------
# 3. Dataset and Collator (fingerprint MLM)
# ---------------------------
class FingerprintDataset(Dataset):
    def __init__(self, fps: List[torch.Tensor]):
        self.fps = fps

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        # Return the tensor directly (not wrapped in a dict). This avoids mismatches
        # when HF's Trainer / collators pass around items in different formats.
        return self.fps[idx]

def collate_batch(batch):
    """
    Collate a batch of fingerprint tensors into:
      - z: [B, L] long, masked/corrupted input tokens (values 0,1, or MASK_TOKEN_ID)
      - labels_z: [B, L] long, with -100 for unselected positions and 0/1 for masked positions (targets)
      - attention_mask: [B, L] bool (all True here since fixed length)

    This collator is defensive: it accepts
      - list of torch.Tensors
      - list of dicts containing key 'fp'
      - HF-style list of dict-like items where a tensor-like value is present
    """
    B = len(batch)
    if B == 0:
        return {"z": torch.zeros((0, FP_LENGTH), dtype=torch.long),
                "labels_z": torch.zeros((0, FP_LENGTH), dtype=torch.long),
                "attention_mask": torch.zeros((0, FP_LENGTH), dtype=torch.bool)}

    # Normalize items -> list of tensors
    tensors = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            # Prefer 'fp' if present
            if "fp" in item:
                val = item["fp"]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.long)
                tensors.append(val)
            else:
                # Try to find any tensor-like value inside dict
                found = None
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        found = v
                        break
                    elif isinstance(v, np.ndarray):
                        found = torch.tensor(v, dtype=torch.long)
                        break
                    elif isinstance(v, list):
                        # possible list of ints
                        try:
                            found = torch.tensor(v, dtype=torch.long)
                            break
                        except Exception:
                            continue
                if found is None:
                    raise KeyError("collate_batch: couldn't find 'fp' tensor in dataset item; item keys: {}".format(list(item.keys())))
                tensors.append(found)
        else:
            # fallback: try to convert numpy/sequence to tensor
            try:
                tensors.append(torch.tensor(item, dtype=torch.long))
            except Exception:
                raise TypeError(f"collate_batch: unsupported batch item type: {type(item)}")

    # Stack into [B, L]
    all_inputs = torch.stack(tensors, dim=0).long()  # [B, L], long (0/1)
    device = all_inputs.device

    # Prepare masks and labels
    labels_z = torch.full_like(all_inputs, fill_value=-100, dtype=torch.long)  # -100 ignored by CE
    z_masked = all_inputs.clone()

    for i in range(B):
        z = all_inputs[i]  # [L]
        n_positions = z.size(0)
        # select positions to supervise (mask) with probability P_MASK
        is_selected = torch.rand(n_positions) < P_MASK

        # ensure not all selected
        if is_selected.all():
            is_selected[torch.randint(0, n_positions, (1,))] = False

        sel_idx = torch.nonzero(is_selected).squeeze(-1)
        if sel_idx.numel() > 0:
            labels_z[i, sel_idx] = z[sel_idx]  # store true bit labels

            # BERT-style corruption per selected position
            probs = torch.rand(sel_idx.size(0))
            mask_choice = probs < 0.8
            rand_choice = (probs >= 0.8) & (probs < 0.9)
            # keep_choice = probs >= 0.9

            if mask_choice.any():
                z_masked[i, sel_idx[mask_choice]] = MASK_TOKEN_ID  # mask token id

            if rand_choice.any():
                # replace with random 0 or 1
                rand_bits = torch.randint(0, 2, (rand_choice.sum().item(),), dtype=torch.long)
                z_masked[i, sel_idx[rand_choice]] = rand_bits

            # keep_choice -> leave original bit

    attention_mask = torch.ones_like(all_inputs, dtype=torch.bool)  # full attention (fixed length)

    return {"z": z_masked, "labels_z": labels_z, "attention_mask": attention_mask}

train_dataset = FingerprintDataset(train_fps)
val_dataset   = FingerprintDataset(val_fps)
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_batch, drop_last=False)
val_loader   = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate_batch, drop_last=False)

# ---------------------------
# 4. Model Definition (Fingerprint Encoder + MLM head)
# ---------------------------

class FingerprintEncoder(nn.Module):
    """
    Simple encoder for fingerprint token sequences:
      - token embedding (vocab size VOCAB_SIZE)
      - positional embedding
      - Transformer encoder stack
      - returns per-position embeddings [B, L, hidden_dim]
    """
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=FP_LENGTH,
                 num_layers=TRANSFORMER_NUM_LAYERS, nhead=TRANSFORMER_NHEAD, dim_feedforward=TRANSFORMER_FF,
                 dropout=DROPOUT):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L] long (values 0,1, or MASK_TOKEN_ID)
        attention_mask: [B, L] bool (True for valid positions)
        returns: embeddings [B, L, hidden_dim]
        """
        B, L = input_ids.shape
        x = self.token_emb(input_ids)  # [B, L, hidden]
        # positional indices 0..L-1 broadcast to batch
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)
        # transformer expects batch_first=True (we set that)
        if attention_mask is not None:
            # transformer encoder in PyTorch doesn't use attention_mask in same way as HF; provide key_padding_mask
            key_padding_mask = ~attention_mask  # True where to mask
        else:
            key_padding_mask = None

        out = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return out  # [B, L, hidden_dim]


class MaskedFingerprintModel(nn.Module):
    """
    Encoder + MLM head for fingerprint masked language modeling.
    MLM head predicts over VOCAB_SIZE (0,1,MASK) like a token classification over the small vocab.
    Loss is standard CrossEntropyLoss (ignore_index=-100) computed only on masked positions,
    matching the "MLM with CrossEntropy" behavior used in the DebertaV2ForMaskedLM setup.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder = FingerprintEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
        # MLM head: predict logits over the small token vocabulary {0,1,MASK}
        self.mlm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, attention_mask=None, labels_z=None):
        """
        z: [B, L] long inputs (0/1/MASK_TOKEN_ID)
        labels_z: [B, L] long with -100 for unselected positions, else 0/1 targets
        Returns:
          - if labels_z provided -> loss (scalar tensor)
          - else -> logits [B, L, VOCAB_SIZE]
        """
        embeddings = self.encoder(z, attention_mask=attention_mask)  # [B, L, hidden]
        logits = self.mlm_head(embeddings)  # [B, L, VOCAB_SIZE]

        if labels_z is not None:
            mask = labels_z != -100  # [B, L]
            if mask.sum() == 0:
                # return zero loss tensor on same device
                return torch.tensor(0.0, device=z.device)

            logits_masked = logits[mask]  # [M, VOCAB_SIZE]
            labels_masked = labels_z[mask]  # [M] values in {0,1}

            # standard cross-entropy over the vocabulary (no class weighting, matching previous Deberta MLM behavior)
            # labels_masked must be long
            labels_masked = labels_masked.long()
            loss_z = F.cross_entropy(logits_masked, labels_masked)

            return loss_z

        # inference -> return logits
        return logits

# instantiate model using MLM-style head and standard cross-entropy loss (no learned weighting/class-weights)
model = MaskedFingerprintModel(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# 5. Training Setup (Hugging Face Trainer)
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
eval_accumulation_steps=1000,   gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_strategy="epoch",
    logging_steps=500,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=torch.cuda.is_available(),
    save_strategy="no",            # callback will save best model
    disable_tqdm=False,
    logging_first_step=True,
    report_to=[],
    # NOTE: set to 0 to avoid DataLoader worker pickling/collate issues in some environments.
    dataloader_num_workers=0,
)

class ValLossCallback(TrainerCallback):
    def __init__(self, trainer_ref=None):
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.patience = 10
        self.best_epoch = None
        self.trainer_ref = trainer_ref

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = int(state.epoch)
        train_loss = next((x["loss"] for x in reversed(state.log_history) if "loss" in x), None)
        print(f"\n=== Epoch {epoch_num}/{args.num_train_epochs} ===")
        if train_loss is not None:
            print(f"Train Loss: {train_loss:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_num = int(state.epoch) + 1

        if self.trainer_ref is None:
            print(f"[Eval] Epoch {epoch_num} - metrics (trainer_ref missing): {metrics}")
            return

        metric_val_loss = None
        if metrics is not None:
            metric_val_loss = metrics.get("eval_loss")

        model_eval = self.trainer_ref.model
        model_eval.eval()

        device_local = next(model_eval.parameters()).device if any(p.numel() > 0 for p in model_eval.parameters()) else torch.device("cpu")

        preds_bits = []
        true_bits = []
        total_loss = 0.0
        n_batches = 0

        logits_masked_list = []
        labels_masked_list = []

        with torch.no_grad():
            for batch in val_loader:
                z = batch["z"].to(device_local)  # [B, L]
                labels_z = batch["labels_z"].to(device_local)
                attention_mask = batch.get("attention_mask", torch.ones_like(z, dtype=torch.bool)).to(device_local)

                # compute loss if possible (model returns scalar loss when labels_z provided)
                try:
                    loss = model_eval(z, attention_mask=attention_mask, labels_z=labels_z)
                except Exception as e:
                    loss = None

                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                    n_batches += 1

                logits = model_eval(z, attention_mask=attention_mask)  # [B, L, VOCAB_SIZE]

                mask = labels_z != -100
                if mask.sum().item() == 0:
                    continue

                logits_masked_list.append(logits[mask])
                labels_masked_list.append(labels_z[mask])

                pred_bits = torch.argmax(logits[mask], dim=-1)
                true_b = labels_z[mask]

                preds_bits.extend(pred_bits.cpu().tolist())
                true_bits.extend(true_b.cpu().tolist())

        avg_val_loss = metric_val_loss if metric_val_loss is not None else ((total_loss / n_batches) if n_batches > 0 else float("nan"))

        accuracy = accuracy_score(true_bits, preds_bits) if len(true_bits) > 0 else 0.0
        f1 = f1_score(true_bits, preds_bits, average="weighted") if len(true_bits) > 0 else 0.0

        # perplexity from masked-token cross-entropy (computed over masked positions only)
        if len(logits_masked_list) > 0:
            all_logits_masked = torch.cat(logits_masked_list, dim=0)
            all_labels_masked = torch.cat(labels_masked_list, dim=0)
            # match previous MLM: standard cross-entropy over the vocabulary
            loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked.long())
            try:
                perplexity = float(torch.exp(loss_z_all).cpu().item())
            except Exception:
                perplexity = float(np.exp(float(loss_z_all.cpu().item())))
        else:
            perplexity = float("nan")

        print(f"\n--- Evaluation after Epoch {epoch_num} ---")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 (weighted): {f1:.4f}")
        print(f"Validation Perplexity (classification head): {perplexity:.4f}")

        # Check for improvement
        if avg_val_loss is not None and not (isinstance(avg_val_loss, float) and np.isnan(avg_val_loss)) and avg_val_loss < self.best_val_loss - 1e-6:
            self.best_val_loss = avg_val_loss
            self.best_epoch = int(state.epoch)
            self.epochs_no_improve = 0
            os.makedirs(BEST_MODEL_DIR, exist_ok=True)
            try:
                torch.save(self.trainer_ref.model.state_dict(), os.path.join(BEST_MODEL_DIR, "pytorch_model.bin"))
                print(f"Saved new best model (epoch {epoch_num}) to {os.path.join(BEST_MODEL_DIR, 'pytorch_model.bin')}")
            except Exception as e:
                print(f"Failed to save best model at epoch {epoch_num}: {e}")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping after {self.patience} epochs with no improvement.")
            control.should_training_stop = True


# Create callback and Trainer
callback = ValLossCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_batch,
    callbacks=[callback]
)
callback.trainer_ref = trainer

# ---------------------------
# 6. Run training
# ---------------------------
start_time = time.time()
trainer.train()
total_time = time.time() - start_time

# ---------------------------
# 7. Final Evaluation (evaluate best saved model on validation set)
# ---------------------------

best_model_path = os.path.join(BEST_MODEL_DIR, "pytorch_model.bin")
if os.path.exists(best_model_path):
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model from {best_model_path}")
    except Exception as e:
        print(f"\nFailed to load best model from {best_model_path}: {e}")

model.eval()
preds_bits_all = []
true_bits_all = []
logits_masked_final = []
labels_masked_final = []

with torch.no_grad():
    for batch in val_loader:
        z = batch["z"].to(device)
        labels_z = batch["labels_z"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(z, dtype=torch.bool)).to(device)

        logits = model(z, attention_mask=attention_mask)  # [B, L, VOCAB_SIZE]

        mask = labels_z != -100
        if mask.sum().item() == 0:
            continue

        logits_masked_final.append(logits[mask])
        labels_masked_final.append(labels_z[mask])

        pred_bits = torch.argmax(logits[mask], dim=-1)
        true_b = labels_z[mask]

        preds_bits_all.extend(pred_bits.cpu().tolist())
        true_bits_all.extend(true_b.cpu().tolist())

accuracy = accuracy_score(true_bits_all, preds_bits_all) if len(true_bits_all) > 0 else 0.0
f1 = f1_score(true_bits_all, preds_bits_all, average="weighted") if len(true_bits_all) > 0 else 0.0

if len(logits_masked_final) > 0:
    all_logits_masked_final = torch.cat(logits_masked_final, dim=0)
    all_labels_masked_final = torch.cat(labels_masked_final, dim=0)
    loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final.long())
    try:
        perplexity_final = float(torch.exp(loss_z_final).cpu().item())
    except Exception:
        perplexity_final = float(np.exp(float(loss_z_final.cpu().item())))
else:
    perplexity_final = float("nan")

best_val_loss = callback.best_val_loss if hasattr(callback, "best_val_loss") else float("nan")
best_epoch_num = (int(callback.best_epoch) + 1) if callback.best_epoch is not None else None

print(f"\n=== Final Results (evaluated on best saved model) ===")
print(f"Total Training Time (s): {total_time:.2f}")
if best_epoch_num is not None:
    print(f"Best Epoch (1-based): {best_epoch_num}")
else:
    print("Best Epoch: (none saved)")

print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1 (weighted): {f1:.4f}")
print(f"Validation Perplexity (classification head): {perplexity_final:.4f}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-trainable Parameters: {non_trainable_params}")
