"""
DeBERTav2.py
DeBERTaV2 masked language modeling pretraining for polymer SMILES (PSMILES).
"""

from __future__ import annotations

import os
import time
import json
import shutil
import argparse
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")


def set_cuda_visible_devices(gpu: str = "0") -> None:
    """Set CUDA_VISIBLE_DEVICES before importing torch/transformers heavy modules."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def parse_args() -> argparse.Namespace:
    """CLI arguments for paths and key training/data settings."""
    parser = argparse.ArgumentParser(description="DeBERTaV2 MLM pretraining for polymer pSMILES.")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/path/to/polymer_structures_unified.csv",
        help="Path to input CSV containing a 'psmiles' column.",
    )
    parser.add_argument("--nrows", type=int, default=5_000_000, help="Number of rows to read from CSV.")
    parser.add_argument(
        "--train_txt",
        type=str,
        default="/path/to/generated_polymer_smiles_5M.txt",
        help="Path to write SentencePiece training text (one SMILES per line).",
    )
    parser.add_argument(
        "--spm_prefix",
        type=str,
        default="/path/to/spm_5M",
        help="SentencePiece model prefix (produces <prefix>.model and <prefix>.vocab).",
    )
    parser.add_argument(
        "--tokenized_dataset_dir",
        type=str,
        default="/path/to/dataset_tokenized_all",
        help="Directory to save/load tokenized HF dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/polybert_output_5M",
        help="Trainer output directory (will contain best/).",
    )
    return parser.parse_args()


def load_psmiles_from_csv(csv_file: str, nrows: int) -> List[str]:
    """Load pSMILES strings from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_file, nrows=nrows, engine="python")
    return df["psmiles"].astype(str).tolist()


def train_val_split(psmiles_list: List[str], test_size: float = 0.2, random_state: int = 42):
    """Split pSMILES into train/val lists."""
    from sklearn.model_selection import train_test_split

    return train_test_split(psmiles_list, test_size=test_size, random_state=random_state)


def write_sentencepiece_training_text(train_psmiles: List[str], train_txt: str) -> None:
    """Write one pSMILES per line for SentencePiece training."""
    os.makedirs(os.path.dirname(os.path.abspath(train_txt)), exist_ok=True)
    with open(train_txt, "w", encoding="utf-8") as f:
        for s in train_psmiles:
            f.write(s.strip() + "\n")


def get_special_tokens() -> List[str]:
    """
    Special tokens + element symbols (upper and lower case) used as user-defined symbols
    for SentencePiece.
    """
    elements = [
        "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn",
        "Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
        "In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At",
        "Rn","Fr","Ra","Ac","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og","Ce","Pr","Nd","Pm",
        "Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"
    ]
    small_elements = [i.lower() for i in elements]

    special_tokens = [
        "<pad>",
        "<mask>",
        "[*]",
        "(", ")", "=", "@", "#",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "-", "+",
        "/", "\\",
        "%", "[", "]",
    ]
    special_tokens += elements + small_elements
    return special_tokens


def train_sentencepiece_if_needed(train_txt: str, spm_model_prefix: str, vocab_size: int = 265) -> str:
    """
    Train SentencePiece model if <prefix>.model does not exist.
    Returns path to the .model file.
    """
    import sentencepiece as spm

    model_path = spm_model_prefix + ".model"
    os.makedirs(os.path.dirname(os.path.abspath(spm_model_prefix)), exist_ok=True)

    if not os.path.isfile(model_path):
        spm.SentencePieceTrainer.train(
            input=train_txt,
            model_prefix=spm_model_prefix,
            vocab_size=vocab_size,
            input_sentence_size=5_000_000,
            character_coverage=1.0,
            user_defined_symbols=get_special_tokens(),
        )
    return model_path


def build_psmiles_tokenizer(spm_path: str, max_len: int = 128):
    """
    Uses SentencePiece-backed DebertaV2Tokenizer.
    """
    from transformers import DebertaV2Tokenizer

    tok = DebertaV2Tokenizer(vocab_file=spm_path, do_lower_case=False)
    tok.add_special_tokens({"pad_token": "<pad>", "mask_token": "<mask>"})
    # store max_len for convenience (not required by HF)
    tok.model_max_length = max_len
    return tok


def tokenize_and_save_dataset(train_psmiles: List[str], val_psmiles: List[str], tokenizer, save_dir: str) -> None:
    """Tokenize train/val and persist the DatasetDict to disk."""
    from datasets import Dataset, DatasetDict

    hf_train = Dataset.from_dict({"text": train_psmiles})
    hf_val = Dataset.from_dict({"text": val_psmiles})

    def tokenize_batch(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=tokenizer.model_max_length)

    train_tok = hf_train.map(tokenize_batch, batched=True, batch_size=10_000, num_proc=10)
    val_tok = hf_val.map(tokenize_batch, batched=True, batch_size=10_000, num_proc=10)

    dataset_dict = DatasetDict({"train": train_tok, "test": val_tok})
    os.makedirs(save_dir, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)


def load_tokenized_dataset(tokenized_dir: str):
    """Load tokenized DatasetDict and set torch formats."""
    from datasets import DatasetDict

    dataset_all = DatasetDict.load_from_disk(tokenized_dir)
    dataset_train = dataset_all["train"]
    dataset_test = dataset_all["test"]

    dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset_train, dataset_test


class EpochMetricsCallback:
    """
    TrainerCallback wrapper that:
    - Tracks best validation loss
    - Implements early stopping on val_loss with patience
    - Saves best model + tokenizer.model copy
    - Prints epoch-level stats
    """

    def __init__(self, tokenizer_model_path: str, output_dir: str, patience: int = 10):
        from transformers.trainer_callback import TrainerCallback
        from sentencepiece import SentencePieceProcessor

        class _CB(TrainerCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def on_epoch_end(self, args, state, control, **kwargs):
                self.outer._on_epoch_end(args, state, control, **kwargs)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                self.outer._on_evaluate(args, state, control, metrics=metrics, **kwargs)

            def on_train_end(self, args, state, control, **kwargs):
                self.outer._on_train_end(args, state, control, **kwargs)

        self._cb_cls = _CB
        self._sp = SentencePieceProcessor()
        self._sp.Load(tokenizer_model_path)

        self.tokenizer_model_path = tokenizer_model_path
        self.output_dir = output_dir

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.patience = patience

        self.all_epochs = []
        self.best_val_f1 = None
        self.best_val_accuracy = None
        self.best_perplexity = None

        self.trainer_ref = None
        self._last_train_loss = None

    def as_trainer_callback(self):
        return self._cb_cls(self)

    def _save_model(self, trainer_obj, suffix: str) -> None:
        if trainer_obj is None:
            return
        model_dir = os.path.join(self.output_dir, suffix)
        os.makedirs(model_dir, exist_ok=True)
        trainer_obj.model.save_pretrained(model_dir)
        try:
            shutil.copyfile(self.tokenizer_model_path, os.path.join(model_dir, "tokenizer.model"))
        except Exception:
            pass

    def _on_epoch_end(self, args, state, control, **kwargs):
        train_loss = None
        for log in reversed(state.log_history):
            if "loss" in log and float(log.get("loss", 0)) != 0.0:
                train_loss = log["loss"]
                break
        self._last_train_loss = train_loss

    def _on_evaluate(self, args, state, control, metrics=None, **kwargs):
        import numpy as np

        eval_metrics = metrics or {}
        eval_loss = eval_metrics.get("eval_loss")
        eval_f1 = eval_metrics.get("eval_f1")
        eval_accuracy = eval_metrics.get("eval_accuracy", None)

        train_loss = self._last_train_loss

        epoch_data = {
            "epoch": state.epoch,
            "train_loss": train_loss,
            "val_loss": eval_loss,
            "val_f1": eval_f1,
            "val_accuracy": eval_accuracy,
            "perplexity": np.exp(eval_loss) if eval_loss is not None else None,
        }
        self.all_epochs.append(epoch_data)

        if eval_loss is not None and eval_loss < self.best_val_loss - 1e-6:
            self.best_val_loss = eval_loss
            self.best_epoch = state.epoch
            self.epochs_no_improve = 0
            self.best_val_f1 = eval_f1
            self.best_val_accuracy = eval_accuracy
            self.best_perplexity = np.exp(eval_loss) if eval_loss is not None else None
            self._save_model(self.trainer_ref, "best")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping: no improvement in val_loss for {self.patience} epochs.")
            control.should_training_stop = True

        total_params = sum(p.numel() for p in self.trainer_ref.model.parameters()) if self.trainer_ref is not None else 0
        trainable_params = sum(p.numel() for p in self.trainer_ref.model.parameters() if p.requires_grad) if self.trainer_ref is not None else 0

        print(f"\n=== Epoch {int(state.epoch)}/{args.num_train_epochs} ===")
        print(f"Train Loss: {train_loss:.4f}" if train_loss is not None else "Train Loss: None")
        print(f"Validation Loss: {eval_loss:.4f}" if eval_loss is not None else "Validation Loss: None")
        print(f"Validation F1: {eval_f1:.4f}" if eval_f1 is not None else "Validation F1: None")
        if eval_accuracy is not None:
            print(f"Validation Accuracy:{eval_accuracy:.4f}")
        if eval_loss is not None:
            print(f"Perplexity: {np.exp(eval_loss):.2f}")
        print(f"Best Val Loss: {self.best_val_loss:.4f} (epoch {int(self.best_epoch)})")
        print(f"Total Params: {total_params}")
        print(f"Trainable Params: {trainable_params}")
        print(f"No improvement count:{self.epochs_no_improve}/{self.patience}")

    def _on_train_end(self, args, state, control, **kwargs):
        print("\n=== Model saved ===")
        print(f"Best model (epoch {int(self.best_epoch)}, val_loss={self.best_val_loss:.4f}): {os.path.join(self.output_dir, 'best')}/")


def compute_metrics(eval_pred):
    """Metrics for MLM: accuracy + weighted F1 computed only on masked (-100 excluded) positions."""
    import numpy as np
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100

    if mask.sum() == 0:
        return {"eval_f1": 0.0, "eval_accuracy": 0.0}

    masked_logits = flat_logits[mask]
    masked_labels = flat_labels[mask]
    preds = np.argmax(masked_logits, axis=-1)

    f1 = f1_score(masked_labels, preds, average="weighted")
    accuracy = float(np.mean(masked_labels == preds))
    return {"eval_f1": f1, "eval_accuracy": accuracy}


# =============================================================================
# Encoder wrapper for MLM training
# =============================================================================

class PSMILESDebertaEncoder:
    """
    Dual-use wrapper:
    - For MLM training (HF Trainer):
        forward(input_ids, attention_mask, labels) -> HF outputs (with .loss, .logits)
    - token_logits(...) helper for reconstruction 
    """

    def __init__(
        self,
        model_dir_or_name: Optional[str] = None,
        hidden_size: int = 600,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 512,
        vocab_size: Optional[int] = None,
        pad_token_id: int = 0,
        emb_dim: int = 600,
    ):
        import torch
        import torch.nn as nn
        from transformers import DebertaV2Config, DebertaV2ForMaskedLM

        self.torch = torch
        self.nn = nn

        if model_dir_or_name is not None:
            self.model = DebertaV2ForMaskedLM.from_pretrained(model_dir_or_name)
        else:
            if vocab_size is None:
                vocab_size = 265  # fallback; will be resized by caller if tokenizer provided
            config = DebertaV2Config(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                intermediate_size=intermediate_size,
                pad_token_id=pad_token_id,
            )
            self.model = DebertaV2ForMaskedLM(config)

        # Use hidden size from config if available
        hs = int(getattr(self.model.config, "hidden_size", hidden_size))
        self.pool_proj = nn.Linear(hs, emb_dim)
        self._device = None

    # ---- nn.Module-like API ----
    def to(self, device):
        self.model.to(device)
        self.pool_proj.to(device)
        self._device = device
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        self.pool_proj.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        for p in self.model.parameters():
            yield p
        for p in self.pool_proj.parameters():
            yield p

    def state_dict(self):
        sd = {"model": self.model.state_dict(), "pool_proj": self.pool_proj.state_dict()}
        return sd

    def load_state_dict(self, state_dict, strict: bool = False):
        if isinstance(state_dict, dict) and "model" in state_dict and "pool_proj" in state_dict:
            self.model.load_state_dict(state_dict["model"], strict=strict)
            self.pool_proj.load_state_dict(state_dict["pool_proj"], strict=strict)
        else:
            # allow loading a raw HF state_dict (best-effort)
            try:
                self.model.load_state_dict(state_dict, strict=strict)
            except Exception:
                # ignore if incompatible
                pass
        return self

    def __call__(self, input_ids, attention_mask=None, labels=None):
        return self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # ---- Core helpers ----
    def _pool_hidden(self, last_hidden_state, attention_mask=None):
        """
        Pool token embeddings -> sequence embedding.
        Use attention-masked mean pooling (robust).
        """
        import torch

        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.to(last_hidden_state.device).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (last_hidden_state * mask).sum(dim=1) / denom
        return pooled

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        If labels is provided -> MLM mode: return HF outputs (Trainer compatible).
        Else -> encoder mode: return pooled embedding.
        """
        if labels is not None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        out = self.model.deberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state
        pooled = self._pool_hidden(last_hidden, attention_mask=attention_mask)
        return self.pool_proj(pooled)

    def token_logits(self, input_ids, attention_mask=None, labels=None):
        """
        - If labels provided: returns loss tensor from HF MLM forward
        - Else: returns token logits (B, L, V)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if labels is not None:
            return outputs.loss
        return outputs.logits


def build_model_and_trainer(tokenizer, dataset_train, dataset_test, spm_model_path: str, output_dir: str):
    """Construct model, training args, callback, and Trainer."""
    import torch
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = PSMILESDebertaEncoder(
        model_dir_or_name=None,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        hidden_size=600,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=512,
        emb_dim=600,
    )
    # resize HF embeddings
    try:
        model.model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=25,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1000,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        logging_first_step=True,
        save_strategy="no",
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to=[],
        disable_tqdm=False,
    )

    callback_wrapper = EpochMetricsCallback(tokenizer_model_path=spm_model_path, output_dir=output_dir, patience=10)

    trainer = Trainer(
        model=model,  # wrapper is Trainer-compatible
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[callback_wrapper.as_trainer_callback()],
    )
    callback_wrapper.trainer_ref = trainer
    return model, trainer, callback_wrapper


def run_training(csv_file: str, nrows: int, train_txt: str, spm_prefix: str, tokenized_dir: str, output_dir: str) -> None:
    """End-to-end: load data, train tokenizer (if needed), tokenize, train model, print final report."""
    psmiles_list = load_psmiles_from_csv(csv_file, nrows=nrows)
    train_psmiles, val_psmiles = train_val_split(psmiles_list, test_size=0.2, random_state=42)

    write_sentencepiece_training_text(train_psmiles, train_txt)
    spm_model_path = train_sentencepiece_if_needed(train_txt, spm_prefix, vocab_size=265)

    tokenizer = build_psmiles_tokenizer(spm_path=spm_model_path, max_len=128)

    tokenize_and_save_dataset(train_psmiles, val_psmiles, tokenizer, tokenized_dir)
    dataset_train, dataset_test = load_tokenized_dataset(tokenized_dir)

    model, trainer, callback = build_model_and_trainer(
        tokenizer=tokenizer,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        spm_model_path=spm_model_path,
        output_dir=output_dir,
    )

    start_time = time.time()
    train_output = trainer.train()
    total_time = time.time() - start_time

    # Final report
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"\n=== Final Results ===")
    print(f"Total Training Time (s): {total_time:.2f}")
    print(f"Best Validation Loss: {callback.best_val_loss:.4f}")
    print(f"Best Validation F1: {callback.best_val_f1:.4f}" if callback.best_val_f1 is not None else "Best Validation F1: None")
    print(f"Best Validation Accuracy: {callback.best_val_accuracy:.4f}" if callback.best_val_accuracy is not None else "Best Validation Accuracy: None")
    print(f"Best Perplexity: {callback.best_perplexity:.2f}" if callback.best_perplexity is not None else "Best Perplexity: None")
    print(f"Best Model Epoch: {int(callback.best_epoch)}")
    try:
        print(f"Final Training Loss: {train_output.training_loss:.4f}")
    except Exception:
        pass
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-trainable Parameters: {non_trainable_params}")


def main():
    args = parse_args()
    set_cuda_visible_devices(args.gpu)

    run_training(
        csv_file=args.csv_file,
        nrows=args.nrows,
        train_txt=args.train_txt,
        spm_prefix=args.spm_prefix,
        tokenized_dir=args.tokenized_dataset_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
