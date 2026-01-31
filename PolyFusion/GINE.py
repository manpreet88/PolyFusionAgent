
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

from transformers import TrainingArguments, Trainer
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

# PyG
from torch_geometric.nn import GINEConv

# ---------------------------
# Configuration / Constants
# ---------------------------
P_MASK = 0.15
# Manual max atomic number (user requested)
MAX_ATOMIC_Z = 85
# Mask token id
MASK_ATOM_ID = MAX_ATOMIC_Z + 1

USE_LEARNED_WEIGHTING = True

# GINE / embedding hyperparams requested
NODE_EMB_DIM = 300   # node embedding dimension
EDGE_EMB_DIM = 300   # edge embedding dimension
NUM_GNN_LAYERS = 5

# Other hyperparams
K_ANCHORS = 6
OUTPUT_DIR = "./gin_output_5M"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data reading settings
csv_path = "polymer_structures_unified_processed.csv"
TARGET_ROWS = 5000000
CHUNKSIZE = 50000

# ---------------------------
# Helper functions
# ---------------------------

def safe_get(d: dict, key: str, default=None):
    return d[key] if (isinstance(d, dict) and key in d) else default

def build_adj_list(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    if edge_index is None or edge_index.numel() == 0:
        return adj
    # edge_index shape [2, E]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        # ensure indices are within range
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adj[u].append(v)
    return adj

def shortest_path_lengths_hops(edge_index, num_nodes):
    """
    Compute all-pairs shortest path lengths in hops (BFS per node).
    Returns an (num_nodes, num_nodes) numpy array with int distances; unreachable -> large number (e.g., num_nodes+1)
    """
    adj = build_adj_list(edge_index, num_nodes)
    INF = num_nodes + 1
    dist_mat = np.full((num_nodes, num_nodes), INF, dtype=np.int32)
    for s in range(num_nodes):
        # BFS
        q = [s]
        dist_mat[s, s] = 0
        head = 0
        while head < len(q):
            u = q[head]; head += 1
            for v in adj[u]:
                if dist_mat[s, v] == INF:
                    dist_mat[s, v] = dist_mat[s, u] + 1
                    q.append(v)
    return dist_mat

def match_edge_attr_to_index(edge_index: torch.Tensor, edge_attr: torch.Tensor, target_dim: int = 3):
    """
    Ensure edge_attr has shape [E_index, D]. Handles common mismatches:
      - If edge_attr is empty/None -> returns zeros of shape [E_index, target_dim].
      - If edge_attr.size(0) == edge_index.size(1) -> return as-is.
      - If edge_attr.size(0) * 2 == edge_index.size(1) -> duplicate (common when features only for undirected edges).
      - Otherwise repeat/truncate edge_attr to match E_index (safe fallback).
    """
    E_idx = edge_index.size(1) if (edge_index is not None and edge_index.numel() > 0) else 0
    if E_idx == 0:
        return torch.zeros((0, target_dim), dtype=torch.float)
    if edge_attr is None or edge_attr.numel() == 0:
        return torch.zeros((E_idx, target_dim), dtype=torch.float)
    E_attr = edge_attr.size(0)
    if E_attr == E_idx:
        # already matches
        if edge_attr.size(1) != target_dim:
            # pad/truncate feature dimension to target_dim
            D = edge_attr.size(1)
            if D < target_dim:
                pad = torch.zeros((E_attr, target_dim - D), dtype=torch.float, device=edge_attr.device)
                return torch.cat([edge_attr, pad], dim=1)
            else:
                return edge_attr[:, :target_dim]
        return edge_attr
    # common case: features provided for undirected edges while edge_index contains both directions
    if E_attr * 2 == E_idx:
        try:
            return torch.cat([edge_attr, edge_attr], dim=0)
        except Exception:
            # fallback to repeat below
            pass
    # fallback: repeat/truncate edge_attr to fit E_idx
    reps = (E_idx + E_attr - 1) // E_attr
    edge_rep = edge_attr.repeat(reps, 1)[:E_idx]
    if edge_rep.size(1) != target_dim:
        D = edge_rep.size(1)
        if D < target_dim:
            pad = torch.zeros((E_idx, target_dim - D), dtype=torch.float, device=edge_rep.device)
            edge_rep = torch.cat([edge_rep, pad], dim=1)
        else:
            edge_rep = edge_rep[:, :target_dim]
    return edge_rep

# ---------------------------
# 1. Load Data from `graph` column (chunked)
# ---------------------------
node_atomic_lists = []
node_chirality_lists = []
node_charge_lists = []
edge_index_lists = []
edge_attr_lists = []
num_nodes_list = []
rows_read = 0

for chunk in pd.read_csv(csv_path, engine="python", chunksize=CHUNKSIZE):
    for idx, row in chunk.iterrows():
        # Prefer 'graph' column JSON (string) per user request
        graph_field = None
        if "graph" in row and not pd.isna(row["graph"]):
            try:
                graph_field = json.loads(row["graph"])
            except Exception:
                # If already parsed or other format
                try:
                    graph_field = row["graph"]
                except Exception:
                    graph_field = None
        else:
            # If no graph column, skip (user requested to use graph column)
            continue

        if graph_field is None:
            continue

        # NODE FEATURES
        node_features = safe_get(graph_field, "node_features", None)
        if not node_features:
            # skip graphs without node_features
            continue

        atomic_nums = []
        chirality_vals = []
        formal_charges = []

        for nf in node_features:
            # atomic number
            an = safe_get(nf, "atomic_num", None)
            if an is None:
                # try alternate keys
                an = safe_get(nf, "atomic_number", 0)
            # chirality (use 0 default)
            ch = safe_get(nf, "chirality", 0)
            # formal charge (use 0 default)
            fc = safe_get(nf, "formal_charge", 0)
            atomic_nums.append(int(an))
            chirality_vals.append(float(ch))
            formal_charges.append(float(fc))

        n_nodes = len(atomic_nums)

        # EDGE INDICES & FEATURES
        edge_indices_raw = safe_get(graph_field, "edge_indices", None)
        edge_features_raw = safe_get(graph_field, "edge_features", None)

        if edge_indices_raw is None:
            # try adjacency_matrix to infer edges
            adj_mat = safe_get(graph_field, "adjacency_matrix", None)
            if adj_mat:
                # adjacency_matrix is list of lists
                srcs = []
                dsts = []
                for i, row_adj in enumerate(adj_mat):
                    for j, val in enumerate(row_adj):
                        if val:
                            srcs.append(i)
                            dsts.append(j)
                edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
                # no edge features available -> create zeros matching edges
                E = edge_index.size(1)
                edge_attr = torch.zeros((E, 3), dtype=torch.float)
            else:
                # no edges found -> skip this graph (GINE requires edges)
                continue
        else:
            # edge_indices_raw expected like [[u,v], [u2,v2], ...] or [[u1,u2,...],[v1,v2,...]]
            if isinstance(edge_indices_raw, list) and len(edge_indices_raw) > 0 and isinstance(edge_indices_raw[0], list):
                # Could be list of pairs or list of lists
                if all(len(pair) == 2 and isinstance(pair[0], int) for pair in edge_indices_raw):
                    # list of pairs
                    srcs = [int(p[0]) for p in edge_indices_raw]
                    dsts = [int(p[1]) for p in edge_indices_raw]
                elif isinstance(edge_indices_raw[0][0], int):
                    # Possibly already in [[srcs],[dsts]] format
                    try:
                        srcs = [int(x) for x in edge_indices_raw[0]]
                        dsts = [int(x) for x in edge_indices_raw[1]]
                    except Exception:
                        # fallback
                        srcs = []
                        dsts = []
                else:
                    srcs = []
                    dsts = []
            else:
                srcs = []
                dsts = []

            if len(srcs) == 0:
                # fallback: skip graph
                continue

            edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

            # Build edge_attr matrix with 3 features: bond_type, stereo, is_conjugated (as float)
            if edge_features_raw and isinstance(edge_features_raw, list):
                bond_types = []
                stereos = []
                is_conjs = []
                for ef in edge_features_raw:
                    bt = safe_get(ef, "bond_type", 0)
                    st = safe_get(ef, "stereo", 0)
                    ic = safe_get(ef, "is_conjugated", False)
                    bond_types.append(float(bt))
                    stereos.append(float(st))
                    is_conjs.append(float(1.0 if ic else 0.0))
                edge_attr = torch.tensor(np.stack([bond_types, stereos, is_conjs], axis=1), dtype=torch.float)
            else:
                # no edge features -> zeros
                E = edge_index.size(1)
                edge_attr = torch.zeros((E, 3), dtype=torch.float)

        # Ensure edge_attr length matches edge_index (fix common mismatches)
        edge_attr = match_edge_attr_to_index(edge_index, edge_attr, target_dim=3)

        # NOTE: we explicitly DO NOT parse or use coordinates (geometry) anywhere.

        # Save lists
        node_atomic_lists.append(torch.tensor(atomic_nums, dtype=torch.long))
        node_chirality_lists.append(torch.tensor(chirality_vals, dtype=torch.float))
        node_charge_lists.append(torch.tensor(formal_charges, dtype=torch.float))
        edge_index_lists.append(edge_index)
        edge_attr_lists.append(edge_attr)
        num_nodes_list.append(n_nodes)

        rows_read += 1
        if rows_read >= TARGET_ROWS:
            break
    if rows_read >= TARGET_ROWS:
        break

if len(node_atomic_lists) == 0:
    raise RuntimeError("No graphs were parsed from the CSV 'graph' column. Check input file and format.")

print(f"Parsed {len(node_atomic_lists)} graphs (using 'graph' column). Using manual max atomic Z = {MAX_ATOMIC_Z}")

# ---------------------------
# 2. Train/Val Split
# ---------------------------
indices = list(range(len(node_atomic_lists)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

def subset(l, idxs):
    return [l[i] for i in idxs]

train_atomic = subset(node_atomic_lists, train_idx)
train_chirality = subset(node_chirality_lists, train_idx)
train_charge = subset(node_charge_lists, train_idx)
train_edge_index = subset(edge_index_lists, train_idx)
train_edge_attr = subset(edge_attr_lists, train_idx)
train_num_nodes = subset(num_nodes_list, train_idx)

val_atomic = subset(node_atomic_lists, val_idx)
val_chirality = subset(node_chirality_lists, val_idx)
val_charge = subset(node_charge_lists, val_idx)
val_edge_index = subset(edge_index_lists, val_idx)
val_edge_attr = subset(edge_attr_lists, val_idx)
val_num_nodes = subset(num_nodes_list, val_idx)

# ---------------------------
# Compute class weights (for weighted CE)
# ---------------------------
num_classes = MASK_ATOM_ID + 1
counts = np.ones((num_classes,), dtype=np.float64)
for z in train_atomic:
    vals = z.cpu().numpy().astype(int)
    for v in vals:
        if 0 <= v < num_classes:
            counts[v] += 1.0
freq = counts / counts.sum()
inv_freq = 1.0 / (freq + 1e-12)
class_weights = inv_freq / inv_freq.mean()
class_weights = torch.tensor(class_weights, dtype=torch.float)
class_weights[MASK_ATOM_ID] = 1.0

# ---------------------------
# 3. Dataset and Collator (build MLM masks + invariant distance targets using hop counts only)
# ---------------------------
class PolymerDataset(Dataset):
    def __init__(self, atomic_list, chirality_list, charge_list, edge_index_list, edge_attr_list, num_nodes_list):
        self.atomic_list = atomic_list
        self.chirality_list = chirality_list
        self.charge_list = charge_list
        self.edge_index_list = edge_index_list
        self.edge_attr_list = edge_attr_list
        self.num_nodes_list = num_nodes_list

    def __len__(self):
        return len(self.atomic_list)

    def __getitem__(self, idx):
        return {
            "z": self.atomic_list[idx],                     # [n_nodes]
            "chirality": self.chirality_list[idx],         # [n_nodes] float
            "formal_charge": self.charge_list[idx],        # [n_nodes]
            "edge_index": self.edge_index_list[idx],       # [2, E]
            "edge_attr": self.edge_attr_list[idx],         # [E, 3]
            "num_nodes": int(self.num_nodes_list[idx])     # int
        }

def collate_batch(batch):
    """
    Builds a batched structure from a list of graph dicts.
    Returns:
      - z: [N_total] long (atomic numbers possibly masked)
      - chirality: [N_total] float
      - formal_charge: [N_total] float
      - edge_index: [2, E_total] long (node indices offset per graph)
      - edge_attr: [E_total, 3] float
      - batch: [N_total] long mapping node->graph idx
      - labels_z: [N_total] long (-100 for unselected)
      - labels_dists: [N_total, K_ANCHORS] float (hop counts)
      - labels_dists_mask: [N_total, K_ANCHORS] bool
    Distance targets:
      - Shortest-path hop distances computed from edge_index for every graph.
    """
    all_z = []
    all_ch = []
    all_fc = []
    all_labels_z = []
    all_labels_dists = []
    all_labels_dists_mask = []
    batch_idx = []
    edge_index_list_batched = []
    edge_attr_list_batched = []
    node_offset = 0
    total_nodes = 0
    total_edges = 0

    for i, g in enumerate(batch):
        z = g["z"]                       # tensor [n]
        n = z.size(0)
        if n == 0:
            continue

        chir = g["chirality"]
        fc = g["formal_charge"]
        edge_index = g["edge_index"]
        edge_attr = g["edge_attr"]

        # Mask selection
        is_selected = torch.rand(n) < P_MASK
        if is_selected.all():
            is_selected[torch.randint(0, n, (1,))] = False

        labels_z = torch.full((n,), -100, dtype=torch.long)
        labels_dists = torch.zeros((n, K_ANCHORS), dtype=torch.float)
        labels_dists_mask = torch.zeros((n, K_ANCHORS), dtype=torch.bool)
        labels_z[is_selected] = z[is_selected]

        # BERT-style corruption on atomic numbers
        z_masked = z.clone()
        if is_selected.any():
            sel_idx = torch.nonzero(is_selected).squeeze(-1)
            rand_atomic = torch.randint(1, MAX_ATOMIC_Z + 1, (sel_idx.size(0),), dtype=torch.long)
            probs = torch.rand(sel_idx.size(0))
            mask_choice = probs < 0.8
            rand_choice = (probs >= 0.8) & (probs < 0.9)
            if mask_choice.any():
                z_masked[sel_idx[mask_choice]] = MASK_ATOM_ID
            if rand_choice.any():
                z_masked[sel_idx[rand_choice]] = rand_atomic[rand_choice]
            # keep_choice -> do nothing

        # Build invariant distance targets using hop distances only
        visible_idx = torch.nonzero(~is_selected).squeeze(-1)
        if visible_idx.numel() == 0:
            visible_idx = torch.arange(n, dtype=torch.long)

        # compute hop distances via BFS using edge_index
        ei = edge_index.clone()
        num_nodes_local = n
        dist_mat = shortest_path_lengths_hops(ei, num_nodes_local)  # numpy int matrix
        for a in torch.nonzero(is_selected).squeeze(-1).tolist():
            # distances to visible nodes
            vis = visible_idx.numpy()
            if vis.size == 0:
                continue
            dists = dist_mat[a, vis].astype(np.float32)
            # filter unreachable (INF = n+1)
            valid_mask = dists <= num_nodes_local
            if not valid_mask.any():
                continue
            dists_valid = dists[valid_mask]
            vis_valid = vis[valid_mask]
            # choose smallest hop distances
            k = min(K_ANCHORS, dists_valid.size)
            idx_sorted = np.argsort(dists_valid)[:k]
            selected_vals = dists_valid[idx_sorted]
            labels_dists[a, :k] = torch.tensor(selected_vals, dtype=torch.float)
            labels_dists_mask[a, :k] = True

        # Append node-level tensors to batched lists
        all_z.append(z_masked)
        all_ch.append(chir)
        all_fc.append(fc)
        all_labels_z.append(labels_z)
        all_labels_dists.append(labels_dists)
        all_labels_dists_mask.append(labels_dists_mask)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))

        # Offset edge indices and append
        if edge_index is not None and edge_index.numel() > 0:
            ei_offset = edge_index + node_offset
            edge_index_list_batched.append(ei_offset)
            # edge_attr already matched earlier; still ensure shapes here for safety
            edge_attr_matched = match_edge_attr_to_index(edge_index, edge_attr, target_dim=3)
            edge_attr_list_batched.append(edge_attr_matched)
            total_edges += edge_index.size(1)

        node_offset += n
        total_nodes += n

    if len(all_z) == 0:
        # Return empty structured batch
        return {
            "z": torch.tensor([], dtype=torch.long),
            "chirality": torch.tensor([], dtype=torch.float),
            "formal_charge": torch.tensor([], dtype=torch.float),
            "edge_index": torch.tensor([[], []], dtype=torch.long),
            "edge_attr": torch.tensor([], dtype=torch.float).reshape(0, 3),
            "batch": torch.tensor([], dtype=torch.long),
            "labels_z": torch.tensor([], dtype=torch.long),
            "labels_dists": torch.tensor([], dtype=torch.float).reshape(0, K_ANCHORS),
            "labels_dists_mask": torch.tensor([], dtype=torch.bool).reshape(0, K_ANCHORS)
        }

    z_batch = torch.cat(all_z, dim=0)
    chir_batch = torch.cat(all_ch, dim=0)
    fc_batch = torch.cat(all_fc, dim=0)
    labels_z_batch = torch.cat(all_labels_z, dim=0)
    labels_dists_batch = torch.cat(all_labels_dists, dim=0)
    labels_dists_mask_batch = torch.cat(all_labels_dists_mask, dim=0)
    batch_batch = torch.cat(batch_idx, dim=0)

    if len(edge_index_list_batched) > 0:
        edge_index_batched = torch.cat(edge_index_list_batched, dim=1)
        edge_attr_batched = torch.cat(edge_attr_list_batched, dim=0)
    else:
        edge_index_batched = torch.tensor([[], []], dtype=torch.long)
        edge_attr_batched = torch.tensor([], dtype=torch.float).reshape(0, 3)

    return {
        "z": z_batch,
        "chirality": chir_batch,
        "formal_charge": fc_batch,
        "edge_index": edge_index_batched,
        "edge_attr": edge_attr_batched,
        "batch": batch_batch,
        "labels_z": labels_z_batch,
        "labels_dists": labels_dists_batch,
        "labels_dists_mask": labels_dists_mask_batch
    }

train_dataset = PolymerDataset(train_atomic, train_chirality, train_charge, train_edge_index, train_edge_attr, train_num_nodes)
val_dataset   = PolymerDataset(val_atomic, val_chirality, val_charge, val_edge_index, val_edge_attr, val_num_nodes)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# ---------------------------
# 4. Model Definition (GINE-based masked model)
# ---------------------------
class GineBlock(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        # MLP used by GINEConv: map (node_dim) -> node_dim
        self.mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.conv = GINEConv(self.mlp)
        self.bn = nn.BatchNorm1d(node_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # GINEConv accepts edge_attr; edge_attr should be same dim as x (or handled in MLP inside)
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.act(x)
        return x

class MaskedGINE(nn.Module):
    def __init__(self,
                 node_emb_dim=NODE_EMB_DIM,
                 edge_emb_dim=EDGE_EMB_DIM,
                 num_layers=NUM_GNN_LAYERS,
                 max_atomic_z=MAX_ATOMIC_Z,
                 class_weights=None):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim
        self.max_atomic_z = max_atomic_z

        # Embedding for atomic numbers (including MASK token)
        num_embeddings = MASK_ATOM_ID + 1
        self.atom_emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=node_emb_dim, padding_idx=None)

        # Small MLP to map numeric node attributes (chirality, formal_charge) -> node_emb_dim
        self.node_attr_proj = nn.Sequential(
            nn.Linear(2, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, node_emb_dim)
        )

        # Edge encoder: maps 3-dim raw edge features -> edge_emb_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim)
        )

        # Project edge_emb -> node_emb_dim if needed (registered in __init__ to avoid dynamic creation)
        if edge_emb_dim != node_emb_dim:
            self._edge_to_node_proj = nn.Linear(edge_emb_dim, node_emb_dim)
        else:
            self._edge_to_node_proj = None

        # GINE layers
        self.gnn_layers = nn.ModuleList([GineBlock(node_emb_dim) for _ in range(num_layers)])

        # Heads
        num_classes_local = MASK_ATOM_ID + 1
        self.atom_head = nn.Linear(node_emb_dim, num_classes_local)
        self.coord_head = nn.Linear(node_emb_dim, K_ANCHORS)

        # Learned uncertainty weighting
        if USE_LEARNED_WEIGHTING:
            self.log_var_z = nn.Parameter(torch.zeros(1))
            self.log_var_pos = nn.Parameter(torch.zeros(1))
        else:
            self.log_var_z = None
            self.log_var_pos = None

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, z, chirality, formal_charge, edge_index, edge_attr,
                batch=None, labels_z=None, labels_dists=None, labels_dists_mask=None):
        """
        z: [N] long (atomic numbers or MASK_ATOM_ID)
        chirality: [N] float
        formal_charge: [N] float
        edge_index: [2, E] long (global batched indices)
        edge_attr: [E, 3] float
        batch: [N] long mapping nodes->graph idx
        labels_*: optional supervision targets as in collate_batch
        """
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # Node embedding
        atom_embedding = self.atom_emb(z)  # [N, node_emb_dim]
        node_attr = torch.stack([chirality, formal_charge], dim=1)  # [N,2]
        node_attr_emb = self.node_attr_proj(node_attr.to(atom_embedding.device))  # [N, node_emb_dim]

        x = atom_embedding + node_attr_emb  # combine categorical and numeric node features

        # Edge embedding
        if edge_attr is None or edge_attr.numel() == 0:
            E = 0 if edge_attr is None else edge_attr.size(0)
            edge_emb = torch.zeros((E, self.edge_emb_dim), dtype=torch.float, device=x.device)
        else:
            edge_emb = self.edge_encoder(edge_attr.to(x.device))  # [E, edge_emb_dim]

        # For GINEConv, edge_attr should match node feature dim used inside GINE (GINE uses the provided nn to process x_j + edge_attr)
        # Project edge_emb -> node_emb_dim if dims differ (registered in __init__)
        if self._edge_to_node_proj is not None:
            edge_for_conv = self._edge_to_node_proj(edge_emb)
        else:
            edge_for_conv = edge_emb

        # Run GNN layers
        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index.to(h.device), edge_for_conv)

        logits = self.atom_head(h)      # [N, num_classes]
        dists_pred = self.coord_head(h)  # [N, K_ANCHORS]

        # Compute loss if labels provided
        if labels_z is not None and labels_dists is not None and labels_dists_mask is not None:
            mask = labels_z != -100
            if mask.sum() == 0:
                return torch.tensor(0.0, device=z.device)

            logits_masked = logits[mask]
            dists_pred_masked = dists_pred[mask]
            labels_z_masked = labels_z[mask]
            labels_dists_masked = labels_dists[mask]
            labels_dists_mask_mask = labels_dists_mask[mask]

            # classification loss
            if self.class_weights is not None:
                loss_z = F.cross_entropy(logits_masked, labels_z_masked.to(logits_masked.device), weight=self.class_weights.to(logits_masked.device))
            else:
                loss_z = F.cross_entropy(logits_masked, labels_z_masked.to(logits_masked.device))

            # distance loss (only where mask true)
            if labels_dists_mask_mask.any():
                preds = dists_pred_masked[labels_dists_mask_mask]
                trues = labels_dists_masked[labels_dists_mask_mask].to(preds.device)
                loss_pos = F.mse_loss(preds, trues, reduction="mean")
            else:
                loss_pos = torch.tensor(0.0, device=z.device)

            if USE_LEARNED_WEIGHTING:
                lz = torch.exp(-self.log_var_z) * loss_z + self.log_var_z
                lp = torch.exp(-self.log_var_pos) * loss_pos + self.log_var_pos
                loss = 0.5 * (lz + lp)
            else:
                alpha = 1.0
                loss = loss_z + alpha * loss_pos

            return loss

        return logits, dists_pred

# Instantiate model
model = MaskedGINE(node_emb_dim=NODE_EMB_DIM,
                   edge_emb_dim=EDGE_EMB_DIM,
                   num_layers=NUM_GNN_LAYERS,
                   max_atomic_z=MAX_ATOMIC_Z,
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
    save_strategy="no",
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
                chir = batch["chirality"].to(device_local)
                fc = batch["formal_charge"].to(device_local)
                edge_index = batch["edge_index"].to(device_local)
                edge_attr = batch["edge_attr"].to(device_local)
                batch_idx = batch["batch"].to(device_local)
                labels_z = batch["labels_z"].to(device_local)
                labels_dists = batch["labels_dists"].to(device_local)
                labels_dists_mask = batch["labels_dists_mask"].to(device_local)

                try:
                    loss = model_eval(z, chir, fc, edge_index, edge_attr, batch_idx, labels_z, labels_dists, labels_dists_mask)
                except Exception as e:
                    loss = None

                if isinstance(loss, torch.Tensor):
                    total_loss += loss.item()
                    n_batches += 1

                logits, dists_pred = model_eval(z, chir, fc, edge_index, edge_attr, batch_idx)

                mask = labels_z != -100
                if mask.sum().item() == 0:
                    continue

                logits_masked_list.append(logits[mask])
                labels_masked_list.append(labels_z[mask])

                pred_z = torch.argmax(logits[mask], dim=-1)
                true_z = labels_z[mask]

                # flatten valid distances
                pred_d = dists_pred[mask][labels_dists_mask[mask]]
                true_d = labels_dists[mask][labels_dists_mask[mask]]

                if pred_d.numel() > 0:
                    pred_dists_all.extend(pred_d.cpu().tolist())
                    true_dists_all.extend(true_d.cpu().tolist())

                preds_z_all.extend(pred_z.cpu().tolist())
                true_z_all.extend(true_z.cpu().tolist())

        avg_val_loss = metric_val_loss if metric_val_loss is not None else ((total_loss / n_batches) if n_batches > 0 else float("nan"))

        accuracy = accuracy_score(true_z_all, preds_z_all) if len(true_z_all) > 0 else 0.0
        f1 = f1_score(true_z_all, preds_z_all, average="weighted") if len(true_z_all) > 0 else 0.0
        rmse = np.sqrt(mean_squared_error(true_dists_all, pred_dists_all)) if len(true_dists_all) > 0 else 0.0
        mae = mean_absolute_error(true_dists_all, pred_dists_all) if len(true_dists_all) > 0 else 0.0

        if len(logits_masked_list) > 0:
            all_logits_masked = torch.cat(logits_masked_list, dim=0)
            all_labels_masked = torch.cat(labels_masked_list, dim=0)
            cw = getattr(model_eval, "class_weights", None)
            if cw is not None:
                cw_device = cw.to(device_local)
                try:
                    loss_z_all = F.cross_entropy(all_logits_masked, all_labels_masked, weight=cw_device)
                except Exception:
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
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation F1 (weighted): {f1:.4f}")
        print(f"Validation RMSE (distances): {rmse:.4f}")
        print(f"Validation MAE  (distances): {mae:.4f}")
        print(f"Validation Perplexity (classification head): {perplexity:.4f}")

        # Save best model by val loss
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
# 7. Final Evaluation (on best saved model)
# ---------------------------
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

logits_masked_list_final = []
labels_masked_list_final = []

with torch.no_grad():
    for batch in val_loader:
        z = batch["z"].to(device)
        chir = batch["chirality"].to(device)
        fc = batch["formal_charge"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        batch_idx = batch["batch"].to(device)
        labels_z = batch["labels_z"].to(device)
        labels_dists = batch["labels_dists"].to(device)
        labels_dists_mask = batch["labels_dists_mask"].to(device)

        logits, dists_pred = model(z, chir, fc, edge_index, edge_attr, batch_idx)

        mask = labels_z != -100
        if mask.sum().item() == 0:
            continue

        logits_masked_list_final.append(logits[mask])
        labels_masked_list_final.append(labels_z[mask])

        pred_z = torch.argmax(logits[mask], dim=-1)
        true_z = labels_z[mask]

        pred_d = dists_pred[mask][labels_dists_mask[mask]]
        true_d = labels_dists[mask][labels_dists_mask[mask]]

        if pred_d.numel() > 0:
            pred_dists_all.extend(pred_d.cpu().tolist())
            true_dists_all.extend(true_d.cpu().tolist())

        preds_z_all.extend(pred_z.cpu().tolist())
        true_z_all.extend(true_z.cpu().tolist())

accuracy = accuracy_score(true_z_all, preds_z_all) if len(true_z_all) > 0 else 0.0
f1 = f1_score(true_z_all, preds_z_all, average="weighted") if len(true_z_all) > 0 else 0.0
rmse = np.sqrt(mean_squared_error(true_dists_all, pred_dists_all)) if len(true_dists_all) > 0 else 0.0
mae = mean_absolute_error(true_dists_all, pred_dists_all) if len(true_dists_all) > 0 else 0.0

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
