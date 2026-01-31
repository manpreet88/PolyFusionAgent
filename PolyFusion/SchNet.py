import os
import json
import time
import shutil

import sys
import csv

# Increase max CSV field size limit
csv.field_size_limit(sys.maxsize)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# PyG (SchNet)
from torch_geometric.nn import SchNet

from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
from torch_geometric.nn import radius_graph

# ---------------------------
# Configuration / Constants
# ---------------------------
P_MASK = 0.15
# NOTE: do NOT infer max atomic number from the dataset; set it manually as requested.
# "At" (Astatine) atomic number = 85 — change this value if your actual maximum differs.
MAX_ATOMIC_Z = 85

# Use a dedicated MASK token index (not 0). We'll place it after the max atomic number.
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

COORD_NOISE_SIGMA = 0.5    # Å (start value, can tune)
USE_LEARNED_WEIGHTING = True

# SchNet hyperparams requested by user:
SCHNET_NUM_GAUSSIANS = 50
SCHNET_NUM_INTERACTIONS = 6
SCHNET_CUTOFF = 10.0      # Å
SCHNET_MAX_NEIGHBORS = 64

# Number of anchor atoms to predict distances to (invariant objective)
K_ANCHORS = 6

# Output directory
OUTPUT_DIR = "./schnet_output_5M"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 1. Load Data (chunked to avoid OOM)
# ---------------------------
csv_path = "Polymer_Foundational_Model/Datasets/polymer_structures_unified_processed.csv"
# target max rows to read (you previously used nrows=2000000)
TARGET_ROWS = 5000000
# choose a chunksize that fits your memory; adjust if needed
CHUNKSIZE = 50000

atomic_lists = []
coord_lists = []
rows_read = 0

# Read in chunks and parse geometry JSON for each chunk to avoid OOM
for chunk in pd.read_csv(csv_path, engine="python", chunksize=CHUNKSIZE):
    # parse geometry column (JSON strings) in this chunk
    geoms_chunk = chunk["geometry"].apply(json.loads)
    for geom in geoms_chunk:
        conf = geom["best_conformer"]
        atomic_lists.append(conf["atomic_numbers"])
        coord_lists.append(conf["coordinates"])

    rows_read += len(chunk)
    if rows_read >= TARGET_ROWS:
        break

# Use manual maximum atomic number (do not compute from data)
max_atomic_z = MAX_ATOMIC_Z
print(f"Using manual max atomic number: {max_atomic_z} (MASK_ATOM_ID={MASK_ATOM_ID})")

# ---------------------------
# 2. Train/Val Split
# ---------------------------
train_idx, val_idx = train_test_split(list(range(len(atomic_lists))), test_size=0.2, random_state=42)
train_z = [torch.tensor(atomic_lists[i], dtype=torch.long) for i in train_idx]
train_pos = [torch.tensor(coord_lists[i], dtype=torch.float) for i in train_idx]
val_z = [torch.tensor(atomic_lists[i], dtype=torch.long) for i in val_idx]
val_pos = [torch.tensor(coord_lists[i], dtype=torch.float) for i in val_idx]

# ---------------------------
# Compute class weights (for weighted CE to mitigate element imbalance)
# ---------------------------
# We create weights for classes [0 .. max_atomic_z, MASK_ATOM_ID] where most labels will be in 1..max_atomic_z.
num_classes = MASK_ATOM_ID + 1  # (0 unused for typical atomic numbers; mask token at end)
counts = np.ones((num_classes,), dtype=np.float64)  # init with 1 to avoid zero division

for z in train_z:
    if z.numel() > 0:
        vals = z.cpu().numpy().astype(int)
        for v in vals:
            if 0 <= v < num_classes:
                counts[v] += 1.0

# Inverse frequency (normalized to mean 1.0)
freq = counts / counts.sum()
inv_freq = 1.0 / (freq + 1e-12)
class_weights = inv_freq / inv_freq.mean()
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Set MASK token weight to 1.0 (it is not used as target in labels_z)
class_weights[MASK_ATOM_ID] = 1.0

# ---------------------------
# 3. Dataset and Collator
# ---------------------------
class PolymerDataset(Dataset):
    def __init__(self, zs, pos_list):
        self.zs = zs
        self.pos_list = pos_list

    def __len__(self):
        return len(self.zs)

    def __getitem__(self, idx):
        return {"z": self.zs[idx], "pos": self.pos_list[idx]}

def collate_batch(batch):
    """
    Masking + create invariant distance targets:
      - Select atoms for masking (P_MASK).
      - For atomic numbers: 80/10/10 BERT-style corruption. Use MASK_ATOM_ID for mask token.
      - For distances: for each masked atom, compute true distances to up to K_ANCHORS visible atoms
        (nearest visible anchors). Produce labels_dists [N, K_ANCHORS] and anchors_exists mask [N, K_ANCHORS].
      - Return labels_z (atomic targets, -100 for unselected) and labels_dists (+ anchors mask).
    """
    all_z = []
    all_pos = []
    all_labels_z = []
    all_labels_dists = []
    all_labels_dists_mask = []
    batch_idx = []

    for i, data in enumerate(batch):
        z = data["z"]            # [n_atoms]
        pos = data["pos"]        # [n_atoms,3]
        n_atoms = z.size(0)
        if n_atoms == 0:
            continue

        # 1) choose which atoms are selected for masking (15%)
        is_selected = torch.rand(n_atoms) < P_MASK

        # ensure not ALL atoms are selected (we need some visible anchors)
        if is_selected.all():
            # set one random atom to unselected
            is_selected[torch.randint(0, n_atoms, (1,))] = False

        # Prepare labels (only for selected atoms)
        labels_z = torch.full((n_atoms,), -100, dtype=torch.long)         # -100 ignored by CE
        # labels_dists: per-atom K distances (0 padded) and mask indicating valid anchors
        labels_dists = torch.zeros((n_atoms, K_ANCHORS), dtype=torch.float)
        labels_dists_mask = torch.zeros((n_atoms, K_ANCHORS), dtype=torch.bool)

        labels_z[is_selected] = z[is_selected]         # true atomic numbers for selecteds

        # 2) apply BERT-style corruption for atomic numbers
        z_masked = z.clone()
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            # sample random atomic numbers from 1..max_atomic_z (avoid 0 which is often unused)
            rand_atomic = torch.randint(1, max_atomic_z + 1, (sel_idx.size(0),), dtype=torch.long)

            probs = torch.rand(sel_idx.size(0))
            mask_choice = probs < 0.8
            rand_choice = (probs >= 0.8) & (probs < 0.9)
            # keep_choice = probs >= 0.9

            if mask_choice.any():
                z_masked[sel_idx[mask_choice]] = MASK_ATOM_ID
            if rand_choice.any():
                z_masked[sel_idx[rand_choice]] = rand_atomic[rand_choice]
            # 10% keep => do nothing

        # 3) coordinate corruption for selected atoms (we still corrupt positions for training robust embeddings)
        pos_masked = pos.clone()
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            probs_c = torch.rand(sel_idx.size(0))
            noisy_choice = probs_c < 0.8
            randpos_choice = (probs_c >= 0.8) & (probs_c < 0.9)

            if noisy_choice.any():
                idx = sel_idx[noisy_choice]
                noise = torch.randn((idx.size(0), 3)) * COORD_NOISE_SIGMA
                pos_masked[idx] = pos_masked[idx] + noise

            if randpos_choice.any():
                idx = sel_idx[randpos_choice]
                mins = pos.min(dim=0).values
                maxs = pos.max(dim=0).values
                randpos = (torch.rand((idx.size(0), 3)) * (maxs - mins)) + mins
                pos_masked[idx] = randpos

        # 4) Build invariant distance targets for masked atoms:
        visible_idx = torch.nonzero(~is_selected).squeeze(-1)
        # If for some reason no visible (shouldn't happen due to earlier guard), fall back to all atoms as visible
        if visible_idx.numel() == 0:
            visible_idx = torch.arange(n_atoms, dtype=torch.long)

        # Precompute pairwise distances
        # pos: [n_atoms,3], visible_pos: [V,3]
        visible_pos = pos[visible_idx]  # true positions for anchors
        for a in torch.nonzero(is_selected).squeeze(-1).tolist():
            # distances from atom a to all visible anchors
            dists = torch.sqrt(((pos[a].unsqueeze(0) - visible_pos) ** 2).sum(dim=1) + 1e-12)
            # find nearest anchors (ascending)
            if dists.numel() > 0:
                k = min(K_ANCHORS, dists.numel())
                nearest_vals, nearest_idx = torch.topk(dists, k, largest=False)
                labels_dists[a, :k] = nearest_vals
                labels_dists_mask[a, :k] = True
            # else leave zeros and mask False

        all_z.append(z_masked)
        all_pos.append(pos_masked)
        all_labels_z.append(labels_z)
        all_labels_dists.append(labels_dists)
        all_labels_dists_mask.append(labels_dists_mask)
        batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))

    if len(all_z) == 0:
        return {"z": torch.tensor([], dtype=torch.long),
                "pos": torch.tensor([], dtype=torch.float).reshape(0, 3),
                "batch": torch.tensor([], dtype=torch.long),
                "labels_z": torch.tensor([], dtype=torch.long),
                "labels_dists": torch.tensor([], dtype=torch.float).reshape(0, K_ANCHORS),
                "labels_dists_mask": torch.tensor([], dtype=torch.bool).reshape(0, K_ANCHORS)}

    z_batch = torch.cat(all_z, dim=0)
    pos_batch = torch.cat(all_pos, dim=0)
    labels_z_batch = torch.cat(all_labels_z, dim=0)
    labels_dists_batch = torch.cat(all_labels_dists, dim=0)
    labels_dists_mask_batch = torch.cat(all_labels_dists_mask, dim=0)
    batch_batch = torch.cat(batch_idx, dim=0)

    return {"z": z_batch, "pos": pos_batch, "batch": batch_batch,
            "labels_z": labels_z_batch,
            "labels_dists": labels_dists_batch,
            "labels_dists_mask": labels_dists_mask_batch}

train_dataset = PolymerDataset(train_z, train_pos)
val_dataset = PolymerDataset(val_z, val_pos)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

from torch_geometric.nn import SchNet as BaseSchNet
from torch_geometric.nn import radius_graph

class NodeSchNet(nn.Module):
    """Custom SchNet that returns node embeddings instead of graph-level predictions"""

    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6,
                 num_gaussians=50, cutoff=10.0, max_num_neighbors=32, readout='add'):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

        # Initialize the base SchNet but we'll only use parts of it
        self.base_schnet = BaseSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout
        )

    def forward(self, z, pos, batch=None):
        """Return node embeddings, not graph-level predictions"""
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # Use the embedding and interaction layers from base SchNet
        h = self.base_schnet.embedding(z)

        # Build edge connectivity
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                 max_num_neighbors=self.max_num_neighbors)

        # Compute edge distances and expand with Gaussians
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.base_schnet.distance_expansion(edge_weight)

        # Apply interaction blocks (message passing)
        for interaction in self.base_schnet.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # STOP HERE - return node embeddings, don't do readout/final layers
        return h  # Shape: [num_nodes, hidden_channels]

# ---------------------------
# 4. Model Definition (SchNet + two heads + learned weighting)
# ---------------------------
class MaskedSchNet(nn.Module):
    def __init__(self,
                 hidden_channels=600,
                 num_interactions=SCHNET_NUM_INTERACTIONS,
                 num_gaussians=SCHNET_NUM_GAUSSIANS,
                 cutoff=SCHNET_CUTOFF,
                 max_atomic_z=max_atomic_z,
                 max_num_neighbors=SCHNET_MAX_NEIGHBORS,
                 class_weights=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.max_atomic_z = max_atomic_z

        # SchNet model from PyG
        self.schnet = NodeSchNet(
    hidden_channels=hidden_channels,
    num_filters=hidden_channels,
    num_interactions=num_interactions,
    num_gaussians=num_gaussians,
    cutoff=cutoff,
    max_num_neighbors=max_num_neighbors
         )

        # Classification head for atomic number (classes 0..max_atomic_z and MASK token)
        num_classes_local = MASK_ATOM_ID + 1
        self.atom_head = nn.Linear(hidden_channels, num_classes_local)

        # Distance-prediction head (predict K_ANCHORS scalar distances per node) -> invariant target
        self.coord_head = nn.Linear(hidden_channels, K_ANCHORS)

        # Learned uncertainty weighting (log-variances) if enabled
        if USE_LEARNED_WEIGHTING:
            self.log_var_z = nn.Parameter(torch.zeros(1))
            self.log_var_pos = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_z = None
            self.log_var_pos = None

        # Class weights for cross entropy
        if class_weights is not None:
            # register as buffer so it moves with .to(device)
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, z, pos, batch, labels_z=None, labels_dists=None, labels_dists_mask=None):
        """
        z: [N] long (atomic numbers or MASK_ATOM_ID)
        pos: [N,3] float (possibly corrupted)
        batch: [N] long (graph indices)
        labels_z: [N] long (-100 for unselected)
        labels_dists: [N, K_ANCHORS] float (0 padded)
        labels_dists_mask: [N, K_ANCHORS] bool (True where anchor exists)
        """
        # Let SchNet produce node embeddings. SchNet builds its own neighbor graph internally.
        # SchNet's forward often accepts (z, pos, batch)
        try:
            h = self.schnet(z=z, pos=pos, batch=batch)
        except TypeError:
            # fallback if different signature
            h = self.schnet(z=z, pos=pos)

        # Node embeddings
        logits = self.atom_head(h)      # [N, num_classes]
        dists_pred = self.coord_head(h)  # [N, K_ANCHORS]

        # If labels provided -> compute loss (aggregated only over masked atoms)
        if labels_z is not None and labels_dists is not None and labels_dists_mask is not None:
            mask = labels_z != -100  # which atoms were selected for supervision
            if mask.sum() == 0:
                # Nothing masked in this batch: return zero loss (avoid NaNs)
                return torch.tensor(0.0, device=z.device)

            logits_masked = logits[mask]                # [M, num_classes]
            dists_pred_masked = dists_pred[mask]        # [M, K_ANCHORS]
            labels_z_masked = labels_z[mask]            # [M]
            labels_dists_masked = labels_dists[mask]    # [M, K_ANCHORS]
            labels_dists_mask_mask = labels_dists_mask[mask]  # [M, K_ANCHORS] bool

            # classification loss (weighted cross entropy)
            if self.class_weights is not None:
                loss_z = F.cross_entropy(logits_masked, labels_z_masked, weight=self.class_weights)
            else:
                loss_z = F.cross_entropy(logits_masked, labels_z_masked)

            # coordinate/distance loss: only over existing anchor distances
            # flatten valid entries
            if labels_dists_mask_mask.any():
                preds = dists_pred_masked[labels_dists_mask_mask]
                trues = labels_dists_masked[labels_dists_mask_mask]
                loss_pos = F.mse_loss(preds, trues, reduction="mean")
            else:
                # no anchor distances present (shouldn't happen), set zero
                loss_pos = torch.tensor(0.0, device=z.device)

            if USE_LEARNED_WEIGHTING:
                lz = torch.exp(-self.log_var_z) * loss_z + self.log_var_z
                lp = torch.exp(-self.log_var_pos) * loss_pos + self.log_var_pos
                loss = 0.5 * (lz + lp)
            else:
                alpha = 1.0
                loss = loss_z + alpha * loss_pos

            return loss

        # Inference: return logits and predicted distances
        return logits, dists_pred

# instantiate model with requested SchNet params and computed class weights
model = MaskedSchNet(hidden_channels=600,
                     num_interactions=SCHNET_NUM_INTERACTIONS,
                     num_gaussians=SCHNET_NUM_GAUSSIANS,
                     cutoff=SCHNET_CUTOFF,
                     max_atomic_z=max_atomic_z,
                     max_num_neighbors=SCHNET_MAX_NEIGHBORS,
                     class_weights=class_weights)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# 5. Training Setup (Hugging Face Trainer)
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    save_strategy="no",            # we will let callback save best model
    disable_tqdm=False,
    logging_first_step=True,
    report_to=[],
    dataloader_num_workers=4,
)

class ValLossCallback(TrainerCallback):
    def __init__(self, trainer_ref=None):
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.patience = 10
        self.best_epoch = None
        self.trainer_ref = trainer_ref

    def on_epoch_end(self, args, state, control, **kwargs):
        # Print epoch starting from 1 instead of 0
        epoch_num = int(state.epoch)
        train_loss = next((x["loss"] for x in reversed(state.log_history) if "loss" in x), None)
        print(f"\n=== Epoch {epoch_num}/{args.num_train_epochs} ===")
        if train_loss is not None:
            print(f"Train Loss: {train_loss:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        When trainer runs evaluation, compute full validation metrics here (accuracy, f1, rmse, mae, perplexity)
        using the provided val_loader and the trainer's model. Save the model when val_loss improves.
        NOTE: Validation loss printed and used for the best-model decision is taken from the `metrics`
        object provided by the Trainer when available, so it matches the Trainer's evaluation output.
        """
        # Compute epoch number for printing (1-based)
        epoch_num = int(state.epoch) + 1

        # If we don't have a trainer reference or val_loader, fallback to printing whatever metrics provided
        if self.trainer_ref is None:
            print(f"[Eval] Epoch {epoch_num} - metrics (trainer_ref missing): {metrics}")
            return

        # If trainer provided an eval_loss in metrics, prefer that value for printing and best-model decision
        metric_val_loss = None
        if metrics is not None:
            metric_val_loss = metrics.get("eval_loss")

        # Evaluate over val_loader to compute other metrics (accuracy, f1, rmse, mae, perplexity)
        model_eval = self.trainer_ref.model
        model_eval.eval()

        device_local = next(model_eval.parameters()).device if any(p.numel() > 0 for p in model_eval.parameters()) else torch.device("cpu")

        preds_z_all = []
        true_z_all = []
        pred_dists_all = []
        true_dists_all = []
        total_loss = 0.0
        n_batches = 0

        logits_masked_list = []
        labels_masked_list = []

        with torch.no_grad():
            for batch in val_loader:
                z = batch["z"].to(device_local)
                pos = batch["pos"].to(device_local)
                batch_idx = batch["batch"].to(device_local)
                labels_z = batch["labels_z"].to(device_local)
                labels_dists = batch["labels_dists"].to(device_local)
                labels_dists_mask = batch["labels_dists_mask"].to(device_local)

                # compute loss using labels (model returns loss when labels provided)
                try:
                    loss = model_eval(z, pos, batch_idx, labels_z, labels_dists, labels_dists_mask)
                except Exception as e:
                    # If model.forward signature is different, skip loss accumulation but still compute preds
                    loss = None

                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                    n_batches += 1

                # inference to get logits and distance preds
                logits, dists_pred = model_eval(z, pos, batch_idx)

                mask = labels_z != -100
                if mask.sum().item() == 0:
                    continue

                # collect masked logits/labels for perplexity
                logits_masked_list.append(logits[mask])
                labels_masked_list.append(labels_z[mask])

                pred_z = torch.argmax(logits[mask], dim=-1)
                true_z = labels_z[mask]

                # flatten valid distances across anchors
                pred_d = dists_pred[mask][labels_dists_mask[mask]]
                true_d = labels_dists[mask][labels_dists_mask[mask]]

                if pred_d.numel() > 0:
                    pred_dists_all.extend(pred_d.cpu().tolist())
                    true_dists_all.extend(true_d.cpu().tolist())

                preds_z_all.extend(pred_z.cpu().tolist())
                true_z_all.extend(true_z.cpu().tolist())

        # If the trainer provided eval_loss, use it; otherwise fall back to the computed average loss
        avg_val_loss = metric_val_loss if metric_val_loss is not None else ((total_loss / n_batches) if n_batches > 0 else float("nan"))

        # Compute metrics (classification + distance regression)
        accuracy = accuracy_score(true_z_all, preds_z_all) if len(true_z_all) > 0 else 0.0
        f1 = f1_score(true_z_all, preds_z_all, average="weighted") if len(true_z_all) > 0 else 0.0
        rmse = np.sqrt(mean_squared_error(true_dists_all, pred_dists_all)) if len(true_dists_all) > 0 else 0.0
        mae = mean_absolute_error(true_dists_all, pred_dists_all) if len(true_dists_all) > 0 else 0.0

        # Compute classification perplexity from masked-token cross-entropy, if available
        if len(logits_masked_list) > 0:
            all_logits_masked = torch.cat(logits_masked_list, dim=0)
            all_labels_masked = torch.cat(labels_masked_list, dim=0)
            # Use model's class_weights if present
            cw = getattr(model_eval, "class_weights", None)
            if cw is not None:
                cw_device = cw.to(device_local)
                try:
                    loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked, weight=cw_device)
                except Exception:
                    # fallback without weight
                    loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked)
            else:
                loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked)
            try:
                perplexity = float(torch.exp(loss_z_all).cpu().item())
            except Exception:
                perplexity = float(np.exp(float(loss_z_all.cpu().item())))
        else:
            perplexity = float("nan")

        print(f"\n--- Evaluation after Epoch {epoch_num} ---")
        # Print validation loss that matches Trainer's evaluation when available
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 (weighted): {f1:.4f}")
        print(f"Validation RMSE (distances): {rmse:.4f}")
        print(f"Validation MAE  (distances): {mae:.4f}")
        print(f"Validation Perplexity (classification head): {perplexity:.4f}")

        # Check for improvement (use a small tolerance)
        if avg_val_loss is not None and not (isinstance(avg_val_loss, float) and np.isnan(avg_val_loss)) and avg_val_loss < self.best_val_loss - 1e-6:
            self.best_val_loss = avg_val_loss
            self.best_epoch = int(state.epoch)  # store 0-based internally
            self.epochs_no_improve = 0
            # Save best model state_dict
            os.makedirs(BEST_MODEL_DIR, exist_ok=True)
            try:
                # Prefer trainer's model (which may be wrapped)
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
# attach trainer_ref so callback can save model
callback.trainer_ref = trainer

# ---------------------------
# 6. Run training
# ---------------------------
start_time = time.time()
trainer.train()
total_time = time.time() - start_time

# ---------------------------
# 7. Final Evaluation (metrics computed on masked atoms in validation set)
#     -> NOTE: per request, we will evaluate the best-saved model (by least val loss)
# ---------------------------
# If a best model was saved by the callback, load it
best_model_path = os.path.join(BEST_MODEL_DIR, "pytorch_model.bin")
if os.path.exists(best_model_path):
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model from {best_model_path}")
    except Exception as e:
        print(f"\nFailed to load best model from {best_model_path}: {e}")

model.eval()
preds_z_all = []
true_z_all = []
pred_dists_all = []
true_dists_all = []

# For computing perplexity in final eval
logits_masked_list_final = []
labels_masked_list_final = []

with torch.no_grad():
    for batch in val_loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        labels_z = batch["labels_z"].to(device)
        labels_dists = batch["labels_dists"].to(device)
        labels_dists_mask = batch["labels_dists_mask"].to(device)

        logits, dists_pred = model(z, pos, batch_idx)  # inference mode returns (logits, dists_pred)

        mask = labels_z != -100
        if mask.sum().item() == 0:
            continue

        # collect masked logits/labels for perplexity
        logits_masked_list_final.append(logits[mask])
        labels_masked_list_final.append(labels_z[mask])

        pred_z = torch.argmax(logits[mask], dim=-1)
        true_z = labels_z[mask]

        # flatten valid distances across anchors
        pred_d = dists_pred[mask][labels_dists_mask[mask]]
        true_d = labels_dists[mask][labels_dists_mask[mask]]

        if pred_d.numel() > 0:
            pred_dists_all.extend(pred_d.cpu().tolist())
            true_dists_all.extend(true_d.cpu().tolist())

        preds_z_all.extend(pred_z.cpu().tolist())
        true_z_all.extend(true_z.cpu().tolist())

# Compute metrics (classification + distance regression)
accuracy = accuracy_score(true_z_all, preds_z_all) if len(true_z_all) > 0 else 0.0
f1 = f1_score(true_z_all, preds_z_all, average="weighted") if len(true_z_all) > 0 else 0.0
rmse = np.sqrt(mean_squared_error(true_dists_all, pred_dists_all)) if len(true_dists_all) > 0 else 0.0
mae = mean_absolute_error(true_dists_all, pred_dists_all) if len(true_dists_all) > 0 else 0.0

# Compute perplexity from collected masked logits/labels
if len(logits_masked_list_final) > 0:
    all_logits_masked_final = torch.cat(logits_masked_list_final, dim=0)
    all_labels_masked_final = torch.cat(labels_masked_list_final, dim=0)
    cw_final = getattr(model, "class_weights", None)
    if cw_final is not None:
        try:
            loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final, weight=cw_final.to(device))
        except Exception:
            loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final)
    else:
        loss_z_final = F.cross_entropy(all_logits_masked_final, all_labels_masked_final)
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
print(f"Validation RMSE (distances): {rmse:.4f}")
print(f"Validation MAE  (distances): {mae:.4f}")
print(f"Validation Perplexity (classification head): {perplexity_final:.4f}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params
print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-trainable Parameters: {non_trainable_params}")
