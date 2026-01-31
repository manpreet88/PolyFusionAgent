import os
import sys
import csv
import json
import time
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Increase csv field size limit safely
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Shared model utilities
from GINE import GineEncoder, match_edge_attr_to_index, safe_get
from SchNet import NodeSchNetWrapper
from Transformer import PooledFingerprintEncoder as FingerprintEncoder
from DeBERTav2 import PSMILESDebertaEncoder, build_psmiles_tokenizer

# HF Trainer & Transformers
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

# ---------------------------
# Config / Hyperparams (paths are placeholders; update for your environment)
# ---------------------------
P_MASK = 0.15
MAX_ATOMIC_Z = 85
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

# GINE params
NODE_EMB_DIM = 300
EDGE_EMB_DIM = 300
NUM_GNN_LAYERS = 5

# SchNet params 
SCHNET_NUM_GAUSSIANS = 50
SCHNET_NUM_INTERACTIONS = 6
SCHNET_CUTOFF = 10.0
SCHNET_MAX_NEIGHBORS = 64
SCHNET_HIDDEN = 600

# Transformer params
FP_LENGTH = 2048
MASK_TOKEN_ID_FP = 2  
VOCAB_SIZE_FP = 3

# DeBERTav2 params 
DEBERTA_HIDDEN = 600
PSMILES_MAX_LEN = 128

# Contrastive params
TEMPERATURE = 0.07

# Reconstruction loss weight (balance between contrastive and reconstruction objectives)
REC_LOSS_WEIGHT = 1.0  # could be tuned (e.g., 0.5, 1.0)

# Training args 
OUTPUT_DIR = "/path/to/multimodal_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_GINE_DIR = "/path/to/gin_output/best"
BEST_SCHNET_DIR = "/path/to/schnet_output/best"
BEST_FP_DIR = "/path/to/fingerprint_mlm_output/best"
BEST_PSMILES_DIR = "/path/to/polybert_output/best"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    weight_decay=0.01,
    eval_accumulation_steps=1000,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    save_steps=500,
    disable_tqdm=False,
    logging_first_step=True,
    report_to=[],
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ======== robust device selection =========
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda")  # respects CUDA_VISIBLE_DEVICES
else:
    device = torch.device("cpu")
print("Device:", device)

# ======== deterministic seeds =========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
#  Utility / small helpers
# ---------------------------

# Optimized BFS to compute distances to visible anchors
def bfs_distances_to_visible(edge_index: torch.Tensor, num_nodes: int, masked_idx: np.ndarray, visible_idx: np.ndarray, k_anchors: int):
    INF = num_nodes + 1
    selected_dists = np.zeros((num_nodes, k_anchors), dtype=np.float32)
    selected_mask = np.zeros((num_nodes, k_anchors), dtype=np.bool_)
    if edge_index is None or edge_index.numel() == 0:
        return selected_dists, selected_mask
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
    visible_set = set(visible_idx.tolist()) if isinstance(visible_idx, (np.ndarray, list)) else set(visible_idx.cpu().tolist())
    for a in np.atleast_1d(masked_idx).tolist():
        if a < 0 or a >= num_nodes:
            continue
        q = [a]
        visited = [-1] * num_nodes
        visited[a] = 0
        head = 0
        found = []
        while head < len(q) and len(found) < k_anchors:
            u = q[head]; head += 1
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    q.append(v)
                    if v in visible_set:
                        found.append((visited[v], v))
                        if len(found) >= k_anchors:
                            break
        if len(found) > 0:
            found.sort(key=lambda x: x[0])
            k = min(k_anchors, len(found))
            for i in range(k):
                selected_dists[a, i] = float(found[i][0])
                selected_mask[a, i] = True
    return selected_dists, selected_mask

# ---------------------------
#  Data loading / preprocessing
# ---------------------------
CSV_PATH = "/path/to/polymer_structures_unified_processed.csv"
TARGET_ROWS = 2000000
CHUNKSIZE = 50000

PREPROC_DIR = "/path/to/preprocessed_samples"
os.makedirs(PREPROC_DIR, exist_ok=True)

# The per-sample file format: torch.save(sample_dict, sample_path)
# sample_dict keys: 'gine', 'schnet', 'fp', 'psmiles_raw'
#   'gine' -> dict with: node_atomic (list/int tensor), chirality (list/float tensor), formal_charge (list/float tensor), edge_index (2xE list), edge_attr (E x 3 list)
#   'schnet' -> dict with: atomic (list), coords (list of [x,y,z])
#   'fp' -> list of length FP_LENGTH (0/1 ints)
#   'psmiles_raw' -> raw psmiles string

def prepare_or_load_data_streaming():
    # If PREPROC_DIR already contains per-sample files, reuse them
    existing = sorted([p for p in Path(PREPROC_DIR).glob("sample_*.pt")])
    if len(existing) > 0:
        print(f"Found {len(existing)} preprocessed sample files in {PREPROC_DIR}; reusing those (no reparse).")
        return [str(p) for p in existing]

    print("No existing per-sample preprocessed folder found. Parsing CSV chunked and writing per-sample files (streaming).")
    rows_read = 0
    sample_idx = 0

    for chunk in pd.read_csv(CSV_PATH, engine="python", chunksize=CHUNKSIZE):
        # Pre-extract columns presence
        has_graph = "graph" in chunk.columns
        has_geometry = "geometry" in chunk.columns
        has_fp = "fingerprints" in chunk.columns
        has_psmiles = "psmiles" in chunk.columns

        for i_row in range(len(chunk)):
            if rows_read >= TARGET_ROWS:
                break
            row = chunk.iloc[i_row]

            # Prepare placeholders
            gine_sample = None
            schnet_sample = None
            fp_sample = None
            psmiles_raw = None

            # Parse graph
            if has_graph:
                val = row.get("graph", "")
                try:
                    graph_field = json.loads(val) if isinstance(val, str) and val.strip() != "" else (val if not isinstance(val, str) else None)
                except Exception:
                    graph_field = None
                if graph_field:
                    node_features = safe_get(graph_field, "node_features", None)
                    if node_features:
                        atomic_nums = []
                        chirality_vals = []
                        formal_charges = []
                        for nf in node_features:
                            an = safe_get(nf, "atomic_num", None)
                            if an is None:
                                an = safe_get(nf, "atomic_number", 0)
                            ch = safe_get(nf, "chirality", 0)
                            fc = safe_get(nf, "formal_charge", 0)
                            try:
                                atomic_nums.append(int(an))
                            except Exception:
                                atomic_nums.append(0)
                            chirality_vals.append(float(ch))
                            formal_charges.append(float(fc))
                        n_nodes = len(atomic_nums)
                        edge_indices_raw = safe_get(graph_field, "edge_indices", None)
                        edge_features_raw = safe_get(graph_field, "edge_features", None)
                        edge_index = None
                        edge_attr = None
                        if edge_indices_raw is None:
                            adj_mat = safe_get(graph_field, "adjacency_matrix", None)
                            if adj_mat:
                                srcs = []
                                dsts = []
                                for i_r, row_adj in enumerate(adj_mat):
                                    for j, val2 in enumerate(row_adj):
                                        if val2:
                                            srcs.append(i_r); dsts.append(j)
                                if len(srcs) > 0:
                                    edge_index = [srcs, dsts]
                                    E = len(srcs)
                                    edge_attr = [[0.0, 0.0, 0.0] for _ in range(E)]
                        else:
                            srcs, dsts = [], []
                            # handle multiple formats
                            if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0 and isinstance(edge_indices_raw[0], list):
                                # either list of pairs or two lists
                                first = edge_indices_raw[0]
                                if len(first) == 2 and isinstance(first[0], int):
                                    # maybe list of pairs
                                    try:
                                        srcs = [int(p[0]) for p in edge_indices_raw]
                                        dsts = [int(p[1]) for p in edge_indices_raw]
                                    except Exception:
                                        srcs, dsts = [], []
                                else:
                                    # maybe [[srcs],[dsts]]
                                    try:
                                        srcs = [int(x) for x in edge_indices_raw[0]]
                                        dsts = [int(x) for x in edge_indices_raw[1]]
                                    except Exception:
                                        srcs, dsts = [], []
                            if len(srcs) == 0 and isinstance(edge_indices_raw, list) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in edge_indices_raw):
                                srcs = [int(p[0]) for p in edge_indices_raw]
                                dsts = [int(p[1]) for p in edge_indices_raw]
                            if len(srcs) > 0:
                                edge_index = [srcs, dsts]
                                if edge_features_raw and isinstance(edge_features_raw, list):
                                    bond_types = []
                                    stereos = []
                                    is_conjs = []
                                    for ef in edge_features_raw:
                                        bt = safe_get(ef, "bond_type", 0)
                                        st = safe_get(ef, "stereo", 0)
                                        ic = safe_get(ef, "is_conjugated", False)
                                        bond_types.append(float(bt)); stereos.append(float(st)); is_conjs.append(float(1.0 if ic else 0.0))
                                    edge_attr = list(zip(bond_types, stereos, is_conjs))
                                else:
                                    E = len(srcs)
                                    edge_attr = [[0.0, 0.0, 0.0] for _ in range(E)]

                        if edge_index is not None:
                            gine_sample = {
                                "node_atomic": atomic_nums,
                                "node_chirality": chirality_vals,
                                "node_charge": formal_charges,
                                "edge_index": edge_index,
                                "edge_attr": edge_attr,
                            }

            # Parse geometry for SchNet
            if has_geometry and schnet_sample is None:
                val = row.get("geometry", "")
                try:
                    geom = json.loads(val) if isinstance(val, str) and val.strip() != "" else (val if not isinstance(val, str) else None)
                    conf = geom.get("best_conformer") if isinstance(geom, dict) else None
                    if conf:
                        atomic = conf.get("atomic_numbers", [])
                        coords = conf.get("coordinates", [])
                        if len(atomic) == len(coords) and len(atomic) > 0:
                            schnet_sample = {"atomic": atomic, "coords": coords}
                except Exception:
                    schnet_sample = None

            # Parse fingerprints
            if has_fp:
                fpval = row.get("fingerprints", "")
                if fpval is None or (isinstance(fpval, str) and fpval.strip() == ""):
                    fp_sample = [0] * FP_LENGTH
                else:
                    try:
                        fp_json = json.loads(fpval) if isinstance(fpval, str) else fpval
                    except Exception:
                        try:
                            fp_json = json.loads(str(fpval).replace("'", '"'))
                        except Exception:
                            parts = [p.strip().strip('"').strip("'") for p in str(fpval).split(",")]
                            bits = [1 if p in ("1", "True", "true") else 0 for p in parts[:FP_LENGTH]]
                            if len(bits) < FP_LENGTH:
                                bits += [0] * (FP_LENGTH - len(bits))
                            fp_sample = bits
                    if fp_sample is None:
                        bits = safe_get(fp_json, "morgan_r3_bits", None) if isinstance(fp_json, dict) else (fp_json if isinstance(fp_json, list) else None)
                        if bits is None:
                            fp_sample = [0] * FP_LENGTH
                        else:
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
                                normalized.extend([0] * (FP_LENGTH - len(normalized)))
                            fp_sample = normalized[:FP_LENGTH]

            # Parse psmiles
            if has_psmiles:
                s = row.get("psmiles", "")
                if s is None:
                    psmiles_raw = ""
                else:
                    psmiles_raw = str(s)

            # If we have at least two modalities (prefer all four), write the sample
            # For safety, we require psmiles and fp at minimum OR graph+psmiles etc.
            modalities_present = sum([1 if x is not None else 0 for x in [gine_sample, schnet_sample, fp_sample, psmiles_raw]])
            if modalities_present >= 2:
                sample = {
                    "gine": gine_sample,
                    "schnet": schnet_sample,
                    "fp": fp_sample,
                    "psmiles_raw": psmiles_raw
                }
                sample_path = os.path.join(PREPROC_DIR, f"sample_{sample_idx:08d}.pt")
                try:
                    torch.save(sample, sample_path)
                except Exception as save_e:
                    print("Warning: failed to torch.save sample:", save_e)
                    # fallback to json write small dict (safe)
                    try:
                        with open(sample_path + ".json", "w") as fjson:
                            json.dump(sample, fjson)
                        # indicate via filename with .json
                        sample_path = sample_path + ".json"
                    except Exception:
                        pass

                sample_idx += 1
                rows_read += 1

            # continue to next row
        if rows_read >= TARGET_ROWS:
            break

    print(f"Wrote {sample_idx} sample files to {PREPROC_DIR}.")
    return [str(p) for p in sorted(Path(PREPROC_DIR).glob("sample_*.pt"))]

sample_files = prepare_or_load_data_streaming()

# ---------------------------
# Prepare tokenizer for psmiles 
# ---------------------------
SPM_MODEL = "/path/to/spm.model"
tokenizer = build_psmiles_tokenizer(spm_path=SPM_MODEL, max_len=PSMILES_MAX_LEN)

# ---------------------------
# Lazy dataset: loads per-sample file on demand and tokenizes psmiles on-the-fly
# ---------------------------
class LazyMultimodalDataset(Dataset):
    def __init__(self, sample_file_list: List[str], tokenizer, fp_length=FP_LENGTH, psmiles_max_len=PSMILES_MAX_LEN):
        self.files = sample_file_list
        self.tokenizer = tokenizer
        self.fp_length = fp_length
        self.psmiles_max_len = psmiles_max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample_path = self.files[idx]
        # prefer torch.load if .pt, else try json
        if sample_path.endswith(".pt"):
            sample = torch.load(sample_path, map_location="cpu")
        else:
            # fallback json load
            with open(sample_path, "r") as f:
                sample = json.load(f)

        # GINE: convert lists to tensors (still on CPU)
        gine_raw = sample.get("gine", None)
        gine_item = None
        if gine_raw:
            node_atomic = torch.tensor(gine_raw.get("node_atomic", []), dtype=torch.long)
            node_chirality = torch.tensor(gine_raw.get("node_chirality", []), dtype=torch.float)
            node_charge = torch.tensor(gine_raw.get("node_charge", []), dtype=torch.float)
            if gine_raw.get("edge_index", None) is not None:
                ei = gine_raw["edge_index"]
                edge_index = torch.tensor(ei, dtype=torch.long)
            else:
                edge_index = torch.tensor([[], []], dtype=torch.long)
            ea_raw = gine_raw.get("edge_attr", None)
            if ea_raw:
                edge_attr = torch.tensor(ea_raw, dtype=torch.float)
            else:
                edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
            gine_item = {"z": node_atomic, "chirality": node_chirality, "formal_charge": node_charge, "edge_index": edge_index, "edge_attr": edge_attr}
        else:
            gine_item = {"z": torch.tensor([], dtype=torch.long), "chirality": torch.tensor([], dtype=torch.float), "formal_charge": torch.tensor([], dtype=torch.float), "edge_index": torch.tensor([[], []], dtype=torch.long), "edge_attr": torch.zeros((0, 3), dtype=torch.float)}

        # SchNet
        schnet_raw = sample.get("schnet", None)
        if schnet_raw:
            s_z = torch.tensor(schnet_raw.get("atomic", []), dtype=torch.long)
            s_pos = torch.tensor(schnet_raw.get("coords", []), dtype=torch.float)
            schnet_item = {"z": s_z, "pos": s_pos}
        else:
            schnet_item = {"z": torch.tensor([], dtype=torch.long), "pos": torch.tensor([], dtype=torch.float)}

        # Fingerprint — stored as list of ints; convert to tensor here
        fp_raw = sample.get("fp", None)
        if fp_raw is None:
            fp_vec = torch.zeros((self.fp_length,), dtype=torch.long)
        else:
            # if fp_raw is already tensor-like, handle it
            if isinstance(fp_raw, (list, tuple)):
                arr = list(fp_raw)[:self.fp_length]
                if len(arr) < self.fp_length:
                    arr = arr + [0] * (self.fp_length - len(arr))
                fp_vec = torch.tensor(arr, dtype=torch.long)
            elif isinstance(fp_raw, torch.Tensor):
                fp_vec = fp_raw.clone().to(torch.long)
            else:
                # fallback
                fp_vec = torch.zeros((self.fp_length,), dtype=torch.long)

        # PSMILES: raw string, tokenize now
        psm_raw = sample.get("psmiles_raw", "")
        if psm_raw is None:
            psm_raw = ""
        enc = self.tokenizer(psm_raw, truncation=True, padding="max_length", max_length=self.psmiles_max_len)
        p_input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        p_attn = torch.tensor(enc["attention_mask"], dtype=torch.bool)

        return {
            "gine": {"z": gine_item["z"], "chirality": gine_item["chirality"], "formal_charge": gine_item["formal_charge"], "edge_index": gine_item["edge_index"], "edge_attr": gine_item["edge_attr"], "num_nodes": int(gine_item["z"].size(0)) if gine_item["z"].numel() > 0 else 0},
            "schnet": {"z": schnet_item["z"], "pos": schnet_item["pos"]},
            "fp": {"input_ids": fp_vec},
            "psmiles": {"input_ids": p_input_ids, "attention_mask": p_attn}
        }

# instantiate dataset lazily
dataset = LazyMultimodalDataset(sample_files, tokenizer, fp_length=FP_LENGTH, psmiles_max_len=PSMILES_MAX_LEN)

# train/val split
train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)

# For manual evaluation (used by evaluate_multimodal), create a DataLoader with num_workers=0
def multimodal_collate(batch_list):
    """
    Given a list of items as returned by MultimodalDataset.__getitem__, build a batched mini-batch
    that the encoders accept.
    """
    B = len(batch_list)
    # GINE batching
    all_z = []
    all_ch = []
    all_fc = []
    all_edge_index = []
    all_edge_attr = []
    batch_mapping = []
    node_offset = 0
    for i, item in enumerate(batch_list):
        g = item["gine"]
        z = g["z"]
        n = z.size(0)
        all_z.append(z)
        all_ch.append(g["chirality"])
        all_fc.append(g["formal_charge"])
        batch_mapping.append(torch.full((n,), i, dtype=torch.long))
        if g["edge_index"] is not None and g["edge_index"].numel() > 0:
            ei_offset = g["edge_index"] + node_offset
            all_edge_index.append(ei_offset)
            ea = match_edge_attr_to_index(g["edge_index"], g["edge_attr"], target_dim=3)
            all_edge_attr.append(ea)
        node_offset += n
    if len(all_z) == 0:
        # create zero-length placeholders for empty batch
        z_batch = torch.tensor([], dtype=torch.long)
        ch_batch = torch.tensor([], dtype=torch.float)
        fc_batch = torch.tensor([], dtype=torch.float)
        batch_batch = torch.tensor([], dtype=torch.long)
        edge_index_batched = torch.empty((2,0), dtype=torch.long)
        edge_attr_batched = torch.zeros((0,3), dtype=torch.float)
    else:
        z_batch = torch.cat(all_z, dim=0)
        ch_batch = torch.cat(all_ch, dim=0)
        fc_batch = torch.cat(all_fc, dim=0)
        batch_batch = torch.cat(batch_mapping, dim=0)
        if len(all_edge_index) > 0:
            edge_index_batched = torch.cat(all_edge_index, dim=1)
            edge_attr_batched = torch.cat(all_edge_attr, dim=0)
        else:
            edge_index_batched = torch.empty((2,0), dtype=torch.long)
            edge_attr_batched = torch.zeros((0,3), dtype=torch.float)

    # SchNet batching: concat nodes and create batch indices
    all_sz = []
    all_pos = []
    schnet_batch = []
    for i, item in enumerate(batch_list):
        s = item["schnet"]
        s_z = s["z"]
        s_pos = s["pos"]
        if s_z.numel() == 0:
            continue
        all_sz.append(s_z)
        all_pos.append(s_pos)
        schnet_batch.append(torch.full((s_z.size(0),), i, dtype=torch.long))
    if len(all_sz) == 0:
        s_z_batch = torch.tensor([], dtype=torch.long)
        s_pos_batch = torch.tensor([], dtype=torch.float)
        s_batch_batch = torch.tensor([], dtype=torch.long)
    else:
        s_z_batch = torch.cat(all_sz, dim=0)
        s_pos_batch = torch.cat(all_pos, dim=0)
        s_batch_batch = torch.cat(schnet_batch, dim=0)

    # FP batching: each fp is vector [L] (long 0/1). We make attention_mask all ones.
    fp_ids = torch.stack([item["fp"]["input_ids"] if isinstance(item["fp"]["input_ids"], torch.Tensor) else torch.tensor(item["fp"]["input_ids"], dtype=torch.long) for item in batch_list], dim=0)
    fp_attn = torch.ones_like(fp_ids, dtype=torch.bool)

    # PSMILES
    p_ids = torch.stack([item["psmiles"]["input_ids"] for item in batch_list], dim=0)
    p_attn = torch.stack([item["psmiles"]["attention_mask"] for item in batch_list], dim=0)

    return {
        "gine": {"z": z_batch, "chirality": ch_batch, "formal_charge": fc_batch, "edge_index": edge_index_batched, "edge_attr": edge_attr_batched, "batch": batch_batch},
        "schnet": {"z": s_z_batch, "pos": s_pos_batch, "batch": s_batch_batch},
        "fp": {"input_ids": fp_ids, "attention_mask": fp_attn},
        "psmiles": {"input_ids": p_ids, "attention_mask": p_attn}
    }

train_loader = DataLoader(train_subset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=multimodal_collate, num_workers=0, drop_last=False)
val_loader = DataLoader(val_subset, batch_size=training_args.per_device_eval_batch_size, shuffle=False, collate_fn=multimodal_collate, num_workers=0, drop_last=False)

# ---------------------------
# Multimodal wrapper & loss 
# ---------------------------
class MultimodalContrastiveModel(nn.Module):
    def __init__(self,
                 gine_encoder: Optional[GineEncoder],
                 schnet_encoder: Optional[NodeSchNetWrapper],
                 fp_encoder: Optional[FingerprintEncoder],
                 psmiles_encoder: Optional[PSMILESDebertaEncoder],
                 emb_dim: int = 600):
        super().__init__()
        self.gine = gine_encoder
        self.schnet = schnet_encoder
        self.fp = fp_encoder
        self.psmiles = psmiles_encoder
        self.proj_gine = nn.Linear(getattr(self.gine, "pool_proj").out_features if self.gine is not None else emb_dim, emb_dim) if self.gine is not None else None
        self.proj_schnet = nn.Linear(getattr(self.schnet, "pool_proj").out_features if self.schnet is not None else emb_dim, emb_dim) if self.schnet is not None else None
        self.proj_fp = nn.Linear(getattr(self.fp, "pool_proj").out_features if self.fp is not None else emb_dim, emb_dim) if self.fp is not None else None
        self.proj_psmiles = nn.Linear(getattr(self.psmiles, "pool_proj").out_features if self.psmiles is not None else emb_dim, emb_dim) if self.psmiles is not None else None
        self.temperature = TEMPERATURE
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def encode(self, batch_mods: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        embs = {}
        B = None
        if 'gine' in batch_mods and self.gine is not None:
            g = batch_mods['gine']
            emb_g = self.gine(g['z'], g['chirality'], g['formal_charge'], g['edge_index'], g['edge_attr'], g.get('batch', None))
            embs['gine'] = F.normalize(self.proj_gine(emb_g), dim=-1)
            B = embs['gine'].size(0) if B is None else B
        if 'schnet' in batch_mods and self.schnet is not None:
            s = batch_mods['schnet']
            emb_s = self.schnet(s['z'], s['pos'], s.get('batch', None))
            embs['schnet'] = F.normalize(self.proj_schnet(emb_s), dim=-1)
            B = embs['schnet'].size(0) if B is None else B
        if 'fp' in batch_mods and self.fp is not None:
            f = batch_mods['fp']
            emb_f = self.fp(f['input_ids'], f.get('attention_mask', None))
            embs['fp'] = F.normalize(self.proj_fp(emb_f), dim=-1)
            B = embs['fp'].size(0) if B is None else B
        if 'psmiles' in batch_mods and self.psmiles is not None:
            p = batch_mods['psmiles']
            emb_p = self.psmiles(p['input_ids'], p.get('attention_mask', None))
            embs['psmiles'] = F.normalize(self.proj_psmiles(emb_p), dim=-1)
            B = embs['psmiles'].size(0) if B is None else B
        return embs

    def forward(self, batch_mods: Dict[str, torch.Tensor], mask_target: str):
        device = next(self.parameters()).device
        embs = self.encode(batch_mods)
        info = {}
        if mask_target not in embs:
            return torch.tensor(0.0, device=device), {"batch_size": 0}
        target = embs[mask_target]
        other_keys = [k for k in embs.keys() if k != mask_target]
        if len(other_keys) == 0:
            return torch.tensor(0.0, device=device), {"batch_size": target.size(0)}
        anchor = torch.stack([embs[k] for k in other_keys], dim=0).mean(dim=0)
        logits = torch.matmul(anchor, target.T) / self.temperature
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        info_nce_loss = F.cross_entropy(logits, labels)
        info['info_nce_loss'] = float(info_nce_loss.detach().cpu().item())

        rec_losses = []
        rec_details = {}

        try:
            if 'gine' in batch_mods and self.gine is not None:
                gm = batch_mods['gine']
                labels_nodes = gm.get('labels', None)
                if labels_nodes is not None:
                    node_logits = self.gine.node_logits(gm['z'], gm['chirality'], gm['formal_charge'], gm['edge_index'], gm['edge_attr'])
                    if labels_nodes.dim() == 1 and node_logits.size(0) == labels_nodes.size(0):
                        loss_gine = self.ce_loss(node_logits, labels_nodes.to(node_logits.device))
                        rec_losses.append(loss_gine)
                        rec_details['gine_rec_loss'] = float(loss_gine.detach().cpu().item())
        except Exception as e:
            print("Warning: GINE reconstruction loss computation failed:", e)

        try:
            if 'schnet' in batch_mods and self.schnet is not None:
                sm = batch_mods['schnet']
                labels_nodes = sm.get('labels', None)
                if labels_nodes is not None:
                    node_logits = self.schnet.node_logits(sm['z'], sm['pos'], sm.get('batch', None))
                    if labels_nodes.dim() == 1 and node_logits.size(0) == labels_nodes.size(0):
                        loss_schnet = self.ce_loss(node_logits, labels_nodes.to(node_logits.device))
                        rec_losses.append(loss_schnet)
                        rec_details['schnet_rec_loss'] = float(loss_schnet.detach().cpu().item())
        except Exception as e:
            print("Warning: SchNet reconstruction loss computation failed:", e)

        try:
            if 'fp' in batch_mods and self.fp is not None:
                fm = batch_mods['fp']
                labels_fp = fm.get('labels', None)
                if labels_fp is not None:
                    token_logits = self.fp.token_logits(fm['input_ids'], fm.get('attention_mask', None))
                    Bf, Lf, V = token_logits.shape
                    logits2 = token_logits.view(-1, V)
                    labels2 = labels_fp.view(-1).to(logits2.device)
                    loss_fp = self.ce_loss(logits2, labels2)
                    rec_losses.append(loss_fp)
                    rec_details['fp_rec_loss'] = float(loss_fp.detach().cpu().item())
        except Exception as e:
            print("Warning: FP reconstruction loss computation failed:", e)

        try:
            if 'psmiles' in batch_mods and self.psmiles is not None:
                pm = batch_mods['psmiles']
                labels_ps = pm.get('labels', None)
                if labels_ps is not None and tokenizer is not None:
                    loss_ps = self.psmiles.token_logits(pm['input_ids'], pm.get('attention_mask', None), labels=labels_ps)
                    if isinstance(loss_ps, torch.Tensor):
                        rec_losses.append(loss_ps)
                        rec_details['psmiles_mlm_loss'] = float(loss_ps.detach().cpu().item())
        except Exception as e:
            print("Warning: PSMILES MLM loss computation failed:", e)

        if len(rec_losses) > 0:
            rec_loss_total = sum(rec_losses) / len(rec_losses)
            info['reconstruction_loss'] = float(rec_loss_total.detach().cpu().item())
            total_loss = info_nce_loss + REC_LOSS_WEIGHT * rec_loss_total
            info['total_loss'] = float(total_loss.detach().cpu().item())
            info.update(rec_details)
        else:
            total_loss = info_nce_loss
            info['reconstruction_loss'] = 0.0
            info['total_loss'] = float(total_loss.detach().cpu().item())

        return total_loss, info

# ---------------------------
# Instantiate encoders (load weights if available) and move to device with .to(device)
gine_encoder = GineEncoder(node_emb_dim=NODE_EMB_DIM, edge_emb_dim=EDGE_EMB_DIM, num_layers=NUM_GNN_LAYERS, max_atomic_z=MAX_ATOMIC_Z)
if os.path.exists(os.path.join(BEST_GINE_DIR, "pytorch_model.bin")):
    try:
        gine_encoder.load_state_dict(torch.load(os.path.join(BEST_GINE_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
        print("Loaded GINE best weights from", BEST_GINE_DIR)
    except Exception as e:
        print("Could not load GINE best weights:", e)
gine_encoder.to(device)

schnet_encoder = NodeSchNetWrapper(hidden_channels=SCHNET_HIDDEN, num_interactions=SCHNET_NUM_INTERACTIONS, num_gaussians=SCHNET_NUM_GAUSSIANS, cutoff=SCHNET_CUTOFF, max_num_neighbors=SCHNET_MAX_NEIGHBORS)
if os.path.exists(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin")):
    try:
        schnet_encoder.load_state_dict(torch.load(os.path.join(BEST_SCHNET_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
        print("Loaded SchNet best weights from", BEST_SCHNET_DIR)
    except Exception as e:
        print("Could not load SchNet best weights:", e)
schnet_encoder.to(device)

fp_encoder = FingerprintEncoder(vocab_size=VOCAB_SIZE_FP, hidden_dim=256, seq_len=FP_LENGTH, num_layers=4, nhead=8, dim_feedforward=1024, dropout=0.1)
if os.path.exists(os.path.join(BEST_FP_DIR, "pytorch_model.bin")):
    try:
        fp_encoder.load_state_dict(torch.load(os.path.join(BEST_FP_DIR, "pytorch_model.bin"), map_location="cpu"), strict=False)
        print("Loaded fingerprint encoder best weights from", BEST_FP_DIR)
    except Exception as e:
        print("Could not load fingerprint best weights:", e)
fp_encoder.to(device)

psmiles_encoder = None
if os.path.isdir(BEST_PSMILES_DIR):
    try:
        psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=BEST_PSMILES_DIR)
        print("Loaded Deberta (PSMILES) from", BEST_PSMILES_DIR)
    except Exception as e:
        print("Failed to load Deberta from saved directory:", e)
if psmiles_encoder is None:
    try:
        psmiles_encoder = PSMILESDebertaEncoder(model_dir_or_name=None)
    except Exception as e:
        print("Failed to instantiate Deberta encoder:", e)
psmiles_encoder.to(device)

multimodal_model = MultimodalContrastiveModel(gine_encoder, schnet_encoder, fp_encoder, psmiles_encoder, emb_dim=600)
multimodal_model.to(device)

# ---------------------------
# Helper to sample masked variant for modalities: (kept same, device-safe)
def mask_batch_for_modality(batch: dict, modality: str, p_mask: float = P_MASK):
    b = {}
    # GINE:
    if 'gine' in batch:
        z = batch['gine']['z'].clone()
        chir = batch['gine']['chirality'].clone()
        fc = batch['gine']['formal_charge'].clone()
        edge_index = batch['gine']['edge_index']
        edge_attr = batch['gine']['edge_attr']
        batch_map = batch['gine'].get('batch', None)
        n_nodes = z.size(0)
        dev = z.device
        is_selected = torch.rand(n_nodes, device=dev) < p_mask
        if is_selected.numel() > 0 and is_selected.all():
            is_selected[torch.randint(0, n_nodes, (1,), device=dev)] = False
        labels_z = torch.full_like(z, fill_value=-100)
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            if sel_idx.dim() == 0:
                sel_idx = sel_idx.unsqueeze(0)
            labels_z[is_selected] = z[is_selected]
            rand_atomic = torch.randint(1, MAX_ATOMIC_Z+1, (sel_idx.size(0),), dtype=torch.long, device=dev)
            probs = torch.rand(sel_idx.size(0), device=dev)
            mask_choice = probs < 0.8
            rand_choice = (probs >= 0.8) & (probs < 0.9)
            if mask_choice.any():
                z[sel_idx[mask_choice]] = MASK_ATOM_ID
            if rand_choice.any():
                z[sel_idx[rand_choice]] = rand_atomic[rand_choice]
        b['gine'] = {"z": z, "chirality": chir, "formal_charge": fc, "edge_index": edge_index, "edge_attr": edge_attr, "batch": batch_map, "labels": labels_z}

    # SchNet:
    if 'schnet' in batch:
        z = batch['schnet']['z'].clone()
        pos = batch['schnet']['pos'].clone()
        batch_map = batch['schnet'].get('batch', None)
        n_nodes = z.size(0)
        dev = z.device
        is_selected = torch.rand(n_nodes, device=dev) < p_mask
        if is_selected.numel() > 0 and is_selected.all():
            is_selected[torch.randint(0, n_nodes, (1,), device=dev)] = False
        labels_z = torch.full((n_nodes,), -100, dtype=torch.long, device=dev)
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            if sel_idx.dim() == 0:
                sel_idx = sel_idx.unsqueeze(0)
            labels_z[is_selected] = z[is_selected]
            probs_c = torch.rand(sel_idx.size(0), device=dev)
            noisy_choice = probs_c < 0.8
            randpos_choice = (probs_c >= 0.8) & (probs_c < 0.9)
            if noisy_choice.any():
                idx = sel_idx[noisy_choice]
                noise = torch.randn((idx.size(0), 3), device=pos.device) * 0.5
                pos[idx] = pos[idx] + noise
            if randpos_choice.any():
                idx = sel_idx[randpos_choice]
                mins = pos.min(dim=0).values
                maxs = pos.max(dim=0).values
                randpos = (torch.rand((idx.size(0), 3), device=pos.device) * (maxs - mins)) + mins
                pos[idx] = randpos
        b['schnet'] = {"z": z, "pos": pos, "batch": batch_map, "labels": labels_z}

    # FP:
    if 'fp' in batch:
        input_ids = batch['fp']['input_ids'].clone()
        attn = batch['fp'].get('attention_mask', torch.ones_like(input_ids, dtype=torch.bool))
        B, L = input_ids.shape
        dev = input_ids.device
        labels_z = torch.full_like(input_ids, -100)
        for i in range(B):
            sel = torch.rand(L, device=dev) < p_mask
            if sel.numel() > 0 and sel.all():
                sel[torch.randint(0, L, (1,), device=dev)] = False
            sel_idx = torch.nonzero(sel).squeeze(-1)
            if sel_idx.numel() > 0:
                if sel_idx.dim() == 0:
                    sel_idx = sel_idx.unsqueeze(0)
                labels_z[i, sel_idx] = input_ids[i, sel_idx]
                probs = torch.rand(sel_idx.size(0), device=dev)
                mask_choice = probs < 0.8
                rand_choice = (probs >= 0.8) & (probs < 0.9)
                if mask_choice.any():
                    input_ids[i, sel_idx[mask_choice]] = MASK_TOKEN_ID_FP
                if rand_choice.any():
                    rand_bits = torch.randint(0, 2, (rand_choice.sum().item(),), dtype=torch.long, device=dev)
                    input_ids[i, sel_idx[rand_choice]] = rand_bits
        b['fp'] = {"input_ids": input_ids, "attention_mask": attn, "labels": labels_z}

    # PSMILES:
    if 'psmiles' in batch:
        input_ids = batch['psmiles']['input_ids'].clone()
        attn = batch['psmiles']['attention_mask'].clone()
        B, L = input_ids.shape
        dev = input_ids.device
        labels_z = torch.full_like(input_ids, -100)
        if tokenizer is None:
            b['psmiles'] = {"input_ids": input_ids, "attention_mask": attn, "labels": labels_z}
        else:
            mask_token_id = tokenizer.mask_token_id if getattr(tokenizer, "mask_token_id", None) is not None else getattr(tokenizer, "vocab", {}).get("<mask>", 1)
            for i in range(B):
                sel = torch.rand(L, device=dev) < p_mask
                if sel.numel() > 0 and sel.all():
                    sel[torch.randint(0, L, (1,), device=dev)] = False
                sel_idx = torch.nonzero(sel).squeeze(-1)
                if sel_idx.numel() > 0:
                    if sel_idx.dim() == 0:
                        sel_idx = sel_idx.unsqueeze(0)
                    labels_z[i, sel_idx] = input_ids[i, sel_idx]
                    probs = torch.rand(sel_idx.size(0), device=dev)
                    mask_choice = probs < 0.8
                    rand_choice = (probs >= 0.8) & (probs < 0.9)
                    if mask_choice.any():
                        input_ids[i, sel_idx[mask_choice]] = mask_token_id
                    if rand_choice.any():
                        rand_ids = torch.randint(0, getattr(tokenizer, "vocab_size", 300), (rand_choice.sum().item(),), dtype=torch.long, device=dev)
                        input_ids[i, sel_idx[rand_choice]] = rand_ids
            b['psmiles'] = {"input_ids": input_ids, "attention_mask": attn, "labels": labels_z}

    return b

def mm_batch_to_model_input(masked_batch):
    mm = {}
    if 'gine' in masked_batch:
        gm = masked_batch['gine']
        mm['gine'] = {"z": gm['z'], "chirality": gm['chirality'], "formal_charge": gm['formal_charge'], "edge_index": gm['edge_index'], "edge_attr": gm['edge_attr'], "batch": gm.get('batch', None), "labels": gm.get('labels', None)}
    if 'schnet' in masked_batch:
        sm = masked_batch['schnet']
        mm['schnet'] = {"z": sm['z'], "pos": sm['pos'], "batch": sm.get('batch', None), "labels": sm.get('labels', None)}
    if 'fp' in masked_batch:
        fm = masked_batch['fp']
        mm['fp'] = {"input_ids": fm['input_ids'], "attention_mask": fm.get('attention_mask', None), "labels": fm.get('labels', None)}
    if 'psmiles' in masked_batch:
        pm = masked_batch['psmiles']
        mm['psmiles'] = {"input_ids": pm['input_ids'], "attention_mask": pm.get('attention_mask', None), "labels": pm.get('labels', None)}
    return mm

# ---------------------------
# Evaluation function (keeps same semantics)
def evaluate_multimodal(model: MultimodalContrastiveModel, val_loader, device, mask_target="fp"):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    acc_sum = 0.0
    top5_sum = 0.0
    mrr_sum = 0.0
    mean_pos_logit_sum = 0.0
    mean_neg_logit_sum = 0.0
    f1_sum = 0.0

    with torch.no_grad():
        for batch in val_loader:
            masked_batch = mask_batch_for_modality(batch, mask_target, p_mask=P_MASK)
            # move to device
            for k in masked_batch:
                for subk in masked_batch[k]:
                    if isinstance(masked_batch[k][subk], torch.Tensor):
                        masked_batch[k][subk] = masked_batch[k][subk].to(device)
            mm_in = mm_batch_to_model_input(masked_batch)
            embs = model.encode(mm_in)
            if mask_target not in embs:
                continue
            target = embs[mask_target]
            other_keys = [k for k in embs.keys() if k != mask_target]
            if len(other_keys) == 0:
                continue
            anchor = torch.stack([embs[k] for k in other_keys], dim=0).mean(dim=0)
            logits = torch.matmul(anchor, target.T) / model.temperature
            B = logits.size(0)
            labels = torch.arange(B, device=logits.device)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * B
            total_examples += B

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            acc_sum += acc * B

            if B >= 5:
                topk = min(5, B)
                topk_indices = torch.topk(logits, k=topk, dim=1).indices
                hits_topk = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
                top5_sum += hits_topk * B
            else:
                top5_sum += acc * B

            sorted_desc = torch.argsort(logits, dim=1, descending=True)
            positions = (sorted_desc == labels.unsqueeze(1)).nonzero(as_tuple=False)
            ranks = torch.zeros(B, device=logits.device).float()
            if positions.numel() > 0:
                for p in positions:
                    i, pos = int(p[0].item()), int(p[1].item())
                    ranks[i] = pos + 1.0
            ranks_nonzero = ranks.clone()
            ranks_nonzero[ranks_nonzero == 0] = float('inf')
            mrr = (1.0 / ranks_nonzero).mean().item()
            mrr_sum += mrr * B

            pos_logits = logits[torch.arange(B), labels]
            neg_logits = logits.clone()
            neg_logits[torch.arange(B), labels] = float('-inf')
            neg_mask = neg_logits != float('-inf')
            if neg_mask.any():
                row_counts = neg_mask.sum(dim=1).clamp(min=1).float()
                sum_neg_per_row = neg_logits.masked_fill(~neg_mask, 0.0).sum(dim=1)
                mean_neg = (sum_neg_per_row / row_counts).mean().item()
            else:
                mean_neg = 0.0
            mean_pos_logit_sum += pos_logits.mean().item() * B
            mean_neg_logit_sum += mean_neg * B

            try:
                labels_np = labels.cpu().numpy()
                preds_np = preds.cpu().numpy()
                if len(np.unique(labels_np)) < 2:
                    batch_f1 = float(acc)
                else:
                    batch_f1 = f1_score(labels_np, preds_np, average='weighted')
            except Exception:
                batch_f1 = float(acc)
            f1_sum += batch_f1 * B

    if total_examples == 0:
        return {"eval_loss": float("nan"), "eval_accuracy": 0.0, "eval_f1_weighted": 0.0}

    avg_loss = total_loss / total_examples
    accuracy = acc_sum / total_examples
    f1_weighted = f1_sum / total_examples

    return {"eval_loss": avg_loss, "eval_accuracy": accuracy, "eval_f1_weighted": f1_weighted}

# ---------------------------
# HF wrapper / Trainer integration (kept same as your part 2, uses lazy loaders)
class HFMultimodalModule(nn.Module):
    def __init__(self, mm_model: MultimodalContrastiveModel):
        super().__init__()
        self.mm = mm_model

    def forward(self, **kwargs):
        if "batch" in kwargs:
            batch = kwargs["batch"]
            mask_target = kwargs.get("mask_target", "fp")
        else:
            modality_keys = ["gine", "schnet", "fp", "psmiles"]
            found = {k: v for k, v in kwargs.items() if k in modality_keys}
            if len(found) > 0:
                batch = {k: found[k] for k in found}
                mask_target = kwargs.get("mask_target", "fp")
            else:
                raise ValueError("HFMultimodalModule.forward could not find 'batch' nor modality keys in inputs. Inputs keys: {}".format(list(kwargs.keys())))
        masked_batch = mask_batch_for_modality(batch, mask_target, p_mask=P_MASK)
        device = next(self.parameters()).device
        for k in masked_batch:
            for subk in list(masked_batch[k].keys()):
                val = masked_batch[k][subk]
                if isinstance(val, torch.Tensor):
                    masked_batch[k][subk] = val.to(device)
        mm_in = mm_batch_to_model_input(masked_batch)
        loss, info = self.mm(mm_in, mask_target)
        logits = None
        labels = None
        try:
            with torch.no_grad():
                embs = self.mm.encode(mm_in)
                if mask_target in embs:
                    target = embs[mask_target]
                    other_keys = [k for k in embs.keys() if k != mask_target]
                    if len(other_keys) > 0:
                        anchor = torch.stack([embs[k] for k in other_keys], dim=0).mean(dim=0)
                        logits = torch.matmul(anchor, target.T) / self.mm.temperature
                        B = logits.size(0)
                        labels = torch.arange(B, device=logits.device)
        except Exception as e:
            print("Warning: failed to compute logits/labels inside HFMultimodalModule.forward:", e)
            logits = None
            labels = None
        eval_loss = loss.detach() if isinstance(loss, torch.Tensor) else torch.tensor(float(loss), device=device)
        out = {"loss": loss, "eval_loss": eval_loss}
        if logits is not None:
            out["logits"] = logits
        if labels is not None:
            out["labels"] = labels
        out["mm_info"] = info
        return out

hf_model = HFMultimodalModule(multimodal_model)
hf_model.to(device)

class ContrastiveDataCollator:
    def __init__(self, mask_prob=P_MASK, modalities: Optional[List[str]] = None):
        self.mask_prob = mask_prob
        self.modalities = modalities if modalities is not None else ["gine", "schnet", "fp", "psmiles"]

    def __call__(self, features):
        if isinstance(features, dict):
            collated = features
            mask_target = random.choice([m for m in self.modalities if m in collated])
            return {"batch": collated, "mask_target": mask_target}
        if isinstance(features, (list, tuple)) and len(features) > 0:
            first = features[0]
            if isinstance(first, dict) and 'gine' in first:
                collated = multimodal_collate(list(features))
                mask_target = random.choice([m for m in self.modalities if m in collated])
                return {"batch": collated, "mask_target": mask_target}
            if isinstance(first, dict) and 'batch' in first:
                collated = first['batch']
                mask_target = first.get("mask_target", random.choice([m for m in self.modalities if m in collated]))
                return {"batch": collated, "mask_target": mask_target}
        print("ContrastiveDataCollator received unexpected 'features' shape/type.")
        raise ValueError("ContrastiveDataCollator could not collate input. Expected list[dict] with 'gine' key or already-collated dict.")

data_collator = ContrastiveDataCollator(mask_prob=P_MASK)

class VerboseTrainingCallback(TrainerCallback):
    def __init__(self, patience: int = 10):
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self._last_train_loss = None
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.patience = patience
        self.trainer_ref = None

    def save_best_model(self, output_dir_suffix: str = "best"):
        if self.trainer_ref is None:
            return
        try:
            ckpt_dir = os.path.join(OUTPUT_DIR, output_dir_suffix)
            os.makedirs(ckpt_dir, exist_ok=True)
            self.trainer_ref._save(ckpt_dir)
            print(f"Saved best model checkpoint to {ckpt_dir}")
        except Exception as e:
            try:
                self.trainer_ref.save_model(os.path.join(OUTPUT_DIR, output_dir_suffix))
                print(f"Saved best model (fallback) to {os.path.join(OUTPUT_DIR, output_dir_suffix)}")
            except Exception as e2:
                print("Warning: failed to save best model:", e, e2)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("" + "="*80)
        print("🚀 STARTING MULTIMODAL CONTRASTIVE LEARNING TRAINING")
        print("="*80)
        model = kwargs.get('model')
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            print(f"📊 MODEL PARAMETERS:")
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Trainable Parameters: {trainable_params:,}")
            print(f"   Non-trainable Parameters: {non_trainable_params:,}")
            print(f"   Training Progress: 0/{args.num_train_epochs} epochs")
        print("="*80)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        current_epoch = state.epoch if state is not None else 0.0
        print(f"📈 Epoch {current_epoch + 1:.1f}/{args.num_train_epochs} Starting...")

    def on_epoch_end(self, args, state, control, **kwargs):
        train_loss = None
        for log in reversed(state.log_history):
            if isinstance(log, dict) and 'loss' in log and float(log.get('loss', 0)) != 0.0:
                train_loss = log['loss']
                break
        self._last_train_loss = train_loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            current_step = state.global_step
            current_epoch = state.epoch
            try:
                steps_per_epoch = max(1, len(train_loader) // args.gradient_accumulation_steps)
            except Exception:
                steps_per_epoch = 1
            if current_step % max(1, steps_per_epoch // 10) == 0:
                progress = current_epoch + (current_step % steps_per_epoch) / steps_per_epoch
                print(f"   Step {current_step:4d} | Epoch {progress:.1f} | Train Loss: {logs['loss']:.6f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_epoch = state.epoch if state is not None else 0.0
        epoch_time = time.time() - self.epoch_start_time
        hf_metrics = metrics if metrics is not None else kwargs.get('metrics', None)
        hf_eval_loss = None
        hf_train_loss = self._last_train_loss
        if hf_metrics is not None:
            hf_eval_loss = hf_metrics.get('eval_loss', hf_metrics.get('loss', None))
            if hf_train_loss is None:
                hf_train_loss = hf_metrics.get('train_loss', hf_train_loss)
        cl_metrics = {}
        try:
            model = kwargs.get('model', None)
            if model is not None:
                cl_model = model.mm if hasattr(model, "mm") else model
                cl_metrics = evaluate_multimodal(cl_model, val_loader, device, mask_target="fp")
            else:
                cl_metrics = evaluate_multimodal(multimodal_model, val_loader, device, mask_target="fp")
        except Exception as e:
            print("Warning: evaluate_multimodal inside callback failed:", e)
        if hf_eval_loss is None:
            hf_eval_loss = cl_metrics.get('eval_loss', None)
        val_acc = cl_metrics.get('eval_accuracy', 'N/A')
        val_f1 = cl_metrics.get('eval_f1_weighted', 'N/A')
        print(f"🔍 EPOCH {current_epoch + 1:.1f} RESULTS:")
        if hf_train_loss is not None:
            try:
                print(f"   Train Loss (HF reported): {hf_train_loss:.6f}")
            except Exception:
                print(f"   Train Loss (HF reported): {hf_train_loss}")
        else:
            print(f"   Train Loss (HF reported): N/A")
        if hf_eval_loss is not None:
            try:
                print(f"   Eval Loss (HF reported): {hf_eval_loss:.6f}")
            except Exception:
                print(f"   Eval Loss (HF reported): {hf_eval_loss}")
        else:
            print(f"   Eval Loss (HF reported): N/A")
        if isinstance(val_acc, float):
            print(f"   Eval Acc (CL evaluator): {val_acc:.6f}")
        else:
            print(f"   Eval Acc (CL evaluator): {val_acc}")
        if isinstance(val_f1, float):
            print(f"   Eval F1 Weighted (CL evaluator): {val_f1:.6f}")
        else:
            print(f"   Eval F1 Weighted (CL evaluator): {val_f1}")
        current_val = hf_eval_loss if hf_eval_loss is not None else float('inf')
        improved = False
        if current_val < self.best_val_loss - 1e-6:
            improved = True
            self.best_val_loss = current_val
            self.best_epoch = current_epoch
            self.epochs_no_improve = 0
            try:
                self.save_best_model("best")
            except Exception as e:
                print("Warning: saving best model failed:", e)
        else:
            self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping: no improvement in val_loss for {self.patience} epochs.")
            control.should_training_stop = True
        print(f"   Epoch Training Time: {epoch_time:.2f}s")
        print(f"   Best Val Loss so far: {self.best_val_loss}")
        print(f"   Epochs since improvement: {self.epochs_no_improve}/{self.patience}")
        print("-" * 50)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("" + "="*80)
        print("🏁 TRAINING COMPLETED")
        print("="*80)
        print(f"   Total Training Time: {total_time:.2f}s")
        if state is not None:
            try:
                print(f"   Total Epochs Completed: {state.epoch + 1:.1f}")
            except Exception:
                pass
        print("="*80)

from transformers import Trainer as HfTrainer

class CLTrainer(HfTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        try:
            metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix) or {}
        except Exception as e:
            print("Warning: super().evaluate() raised an exception. Falling back to CL-only evaluator.")
            import traceback
            traceback.print_exc()
            try:
                cl_model = self.model.mm if hasattr(self.model, "mm") else self.model
                cl_metrics = evaluate_multimodal(cl_model, val_loader, device, mask_target="fp")
                metrics = {k: float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v for k, v in cl_metrics.items()}
                metrics["epoch"] = float(self.state.epoch) if getattr(self.state, "epoch", None) is not None else metrics.get("epoch", 0.0)
            except Exception as e2:
                print("Fallback evaluate_multimodal failed as well:", e2)
                traceback.print_exc()
                metrics = {"eval_loss": float("nan"), "epoch": float(self.state.epoch) if getattr(self.state, "epoch", None) is not None else 0.0}
            return metrics
        try:
            cl_model = self.model.mm if hasattr(self.model, "mm") else self.model
            cl_metrics = evaluate_multimodal(cl_model, val_loader, device, mask_target="fp")
        except Exception as e:
            print("Warning: evaluate_multimodal failed inside CLTrainer.evaluate():", e)
            cl_metrics = {}
        for k, v in cl_metrics.items():
            try:
                metrics[k] = float(v)
            except Exception:
                metrics[k] = v
        if 'eval_loss' not in metrics and 'eval_loss' in cl_metrics:
            try:
                metrics['eval_loss'] = float(cl_metrics['eval_loss'])
            except Exception:
                metrics['eval_loss'] = cl_metrics['eval_loss']
        if "epoch" not in metrics:
            metrics["epoch"] = float(self.state.epoch) if getattr(self.state, "epoch", None) is not None else metrics.get("epoch", 0.0)
        return metrics

    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        try:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        except Exception:
            pass
        try:
            model_to_save = self.model.mm if hasattr(self.model, "mm") else self.model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        except Exception as e:
            try:
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(output_dir)
                else:
                    raise e
            except Exception as e2:
                print("Warning: failed to save model state_dict:", e2)
        try:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        except Exception:
            pass
        try:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        except Exception:
            pass

    def _load_best_model(self):
        best_ckpt = self.state.best_model_checkpoint
        if not best_ckpt:
            return
        candidate = os.path.join(best_ckpt, "pytorch_model.bin")
        if not os.path.exists(candidate):
            candidate = os.path.join(best_ckpt, "model.bin")
            if not os.path.exists(candidate):
                candidate = None
        if candidate is None:
            print(f"CLTrainer._load_best_model(): no compatible pytorch_model.bin found in {best_ckpt}; skipping load.")
            return
        try:
            state_dict = torch.load(candidate, map_location=self.args.device)
            model_to_load = self.model.mm if hasattr(self.model, "mm") else self.model
            model_to_load.load_state_dict(state_dict, strict=False)
            print(f"CLTrainer: loaded best model state_dict from {candidate}")
        except Exception as e:
            print("CLTrainer._load_best_model: failed to load state_dict using torch.load:", e)
            return

callback = VerboseTrainingCallback(patience=10)

trainer = CLTrainer(
    model=hf_model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=val_subset,
    data_collator=data_collator,
    callbacks=[callback],
)

callback.trainer_ref = trainer

# Force HF Trainer to use our prebuilt PyTorch DataLoaders
trainer.get_train_dataloader = lambda dataset=None: train_loader
trainer.get_eval_dataloader  = lambda eval_dataset=None: val_loader

training_args.metric_for_best_model = "eval_loss"
training_args.greater_is_better = False

optimizer = torch.optim.AdamW(multimodal_model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

total_params = sum(p.numel() for p in multimodal_model.parameters())
trainable_params = sum(p.numel() for p in multimodal_model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"\n📊 MODEL PARAMETERS:")
print(f"   Total Parameters: {total_params:,}")
print(f"   Trainable Parameters: {trainable_params:,}")
print(f"   Non-trainable Parameters: {non_trainable_params:,}")

def compute_metrics_wrapper(eval_pred):
    return evaluate_multimodal(multimodal_model, val_loader, device, mask_target="fp")

# ---------------------------
# Clear any cached GPU memory before starting (helpful)
if USE_CUDA:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

# ---------------------------
# Start training
training_start_time = time.time()
trainer.train()
training_end_time = time.time()

# Save best
BEST_MULTIMODAL_DIR = os.path.join(OUTPUT_DIR, "best")
os.makedirs(BEST_MULTIMODAL_DIR, exist_ok=True)

try:
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        multimodal_model.load_state_dict(torch.load(os.path.join(best_ckpt, "pytorch_model.bin"), map_location=device), strict=False)
        print(f"Loaded best checkpoint from {best_ckpt} into multimodal_model for final evaluation.")
    torch.save(multimodal_model.state_dict(), os.path.join(BEST_MULTIMODAL_DIR, "pytorch_model.bin"))
    print(f"✅ Saved best multimodal model to {os.path.join(BEST_MULTIMODAL_DIR, 'pytorch_model.bin')}")
except Exception as e:
    print("Warning: failed to load/save best model from Trainer:", e)

# Final evaluation
final_metrics = {}
try:
    if trainer.state.best_model_checkpoint:
        trainer._load_best_model()
        final_metrics = trainer.evaluate(eval_dataset=val_subset)
    else:
        final_metrics = evaluate_multimodal(multimodal_model, val_loader, device, mask_target="fp")
except Exception as e:
    print("Warning: final evaluation via trainer.evaluate failed, falling back to direct evaluate_multimodal:", e)
    final_metrics = evaluate_multimodal(multimodal_model, val_loader, device, mask_target="fp")

print("\n" + "="*80)
print("🏁 FINAL TRAINING RESULTS")
print("="*80)
training_time = training_end_time - training_start_time
print(f"Total Training Time: {training_time:.2f}s")
best_ckpt = trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None
if best_ckpt:
    print(f"Best Checkpoint: {best_ckpt}")
else:
    print("Best Checkpoint: (none saved)")

hf_eval_loss = final_metrics.get('eval_loss', float('nan'))
hf_eval_acc = final_metrics.get('eval_accuracy', 0.0)
hf_eval_f1 = final_metrics.get('eval_f1_weighted', 0.0)
print(f"Val Loss (HF reported / trainer.evaluate): {hf_eval_loss:.4f}")
print(f"Val Acc (CL evaluator): {hf_eval_acc:.4f}")
print(f"Val F1 Weighted (CL evaluator): {hf_eval_f1:.4f}")
print(f"Total Trainable Params: {trainable_params:,}")
print(f"Total Non-trainable Params: {non_trainable_params:,}")
print("="*80)
