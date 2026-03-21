"""
Multi-LABEL token classification with DDP + Focal Loss.
Each residue can have MULTIPLE labels simultaneously (e.g. "1-2" means label 1 AND label 2).

.npy label format:
  - 2D array (seq_len, num_labels) with 0/1 multi-hot encoding  [preferred]
  - 1D array (seq_len,) with single integers  [auto-converted to one-hot]

Launch examples:
  accelerate launch --num_processes 4 train_Tokens_multilabel_ddp.py \
      --num_labels 3 --loss_type focal --focal_gamma 2.0 --auto_pos_weight ...
  python train_Tokens_multilabel_ddp.py --num_labels 3 ...
"""

import os, sys, re
import argparse
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import torch
import datasets
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from transformers import EsmForTokenClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'


# ---------------------------------------------------------------------------
# Multi-label Focal Loss (per-element binary focal loss)
# ---------------------------------------------------------------------------
class MultilabelFocalLoss(nn.Module):
    """Binary focal loss applied independently to each label."""

    def __init__(self, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits, targets: (N, C)
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device).unsqueeze(0)
            alpha = pw * targets + (1 - targets)
            focal_weight = focal_weight * alpha

        loss = focal_weight * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Utility: convert "1-2" dash-separated labels to multi-hot .npy
# ---------------------------------------------------------------------------
def encode_multilabel(label_strings, num_labels):
    """Convert dash-separated label strings to multi-hot array.

    Args:
        label_strings: list of strings, e.g. ["0", "1", "1-2", "2", "0"]
        num_labels: total number of labels

    Returns:
        np.ndarray of shape (seq_len, num_labels) with 0/1 values

    Usage:
        labels = encode_multilabel(["0", "1-2", "2"], num_labels=3)
        np.save("protein_id.npy", labels)
    """
    seq_len = len(label_strings)
    multi_hot = np.zeros((seq_len, num_labels), dtype=np.int64)
    for i, s in enumerate(label_strings):
        for lab in str(s).split('-'):
            lab_int = int(lab.strip())
            if 0 <= lab_int < num_labels:
                multi_hot[i, lab_int] = 1
    return multi_hot


# ---------------------------------------------------------------------------
# Custom Trainer with BCE / Focal loss for multi-label
# ---------------------------------------------------------------------------
class MultilabelTrainer(Trainer):

    def __init__(self, *args, loss_type='bce', pos_weight=None,
                 focal_gamma=2.0, num_labels=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # (batch, seq_len, num_labels)
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, num_labels)

        device = logits.device
        num_labels = self.num_labels

        # Flatten to (total_tokens, num_labels) and mask out padding (-100)
        logits_flat = logits.reshape(-1, num_labels)
        labels_flat = labels.reshape(-1, num_labels).float()
        mask = (labels_flat[:, 0] != -100)
        logits_valid = logits_flat[mask]
        labels_valid = labels_flat[mask]

        if logits_valid.numel() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        elif self.loss_type == 'focal':
            pw = self.pos_weight.to(device) if self.pos_weight is not None else None
            loss_fct = MultilabelFocalLoss(gamma=self.focal_gamma, pos_weight=pw)
            loss = loss_fct(logits_valid, labels_valid)
        elif self.loss_type == 'bce_weighted':
            pw = self.pos_weight.to(device) if self.pos_weight is not None else None
            loss = F.binary_cross_entropy_with_logits(logits_valid, labels_valid,
                                                       pos_weight=pw)
        else:
            loss = F.binary_cross_entropy_with_logits(logits_valid, labels_valid)

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Custom Data Collator for 2D multi-label labels
# ---------------------------------------------------------------------------
class MultiLabelDataCollator:
    def __init__(self, tokenizer, num_labels, pad_value=-100):
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.pad_value = pad_value

    def __call__(self, features):
        input_features = [{'input_ids': f['input_ids'],
                           'attention_mask': f['attention_mask']} for f in features]
        batch = self.tokenizer.pad(input_features, return_tensors='pt')

        max_len = batch['input_ids'].shape[1]
        padded_labels = []
        for f in features:
            lab = f['labels']
            if isinstance(lab, list):
                lab = np.array(lab, dtype=np.float32)
            if isinstance(lab, np.ndarray):
                lab = torch.from_numpy(lab.astype(np.float32))

            curr_len = lab.shape[0]
            if curr_len < max_len:
                pad = torch.full((max_len - curr_len, self.num_labels),
                                 self.pad_value, dtype=lab.dtype)
                lab = torch.cat([lab, pad], dim=0)
            elif curr_len > max_len:
                lab = lab[:max_len]
            padded_labels.append(lab)

        batch['labels'] = torch.stack(padded_labels)
        return batch


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MyDataset(Dataset):

    def __init__(self, data_table, label_path, num_labels):
        df = pd.read_csv(data_table)
        self.names = df['ProId'].tolist()
        self.sequences = df['Sequence'].tolist()
        self.prolabels = df['Class'].tolist()
        self.label_path = label_path
        self.num_labels = num_labels

    def __getitem__(self, index):
        name = self.names[index]
        sequence = self.sequences[index]
        prolabel = self.prolabels[index]
        seq_len = len(sequence)

        label = np.zeros((seq_len, self.num_labels), dtype=np.float32)

        if prolabel >= 1:
            raw = np.load(os.path.join(self.label_path, name + '.npy'),
                          allow_pickle=True)
            if raw.ndim == 2:
                # Already 2D multi-hot: (seq_len, num_labels)
                label = raw.astype(np.float32)
            elif raw.ndim == 1:
                for i, v in enumerate(raw):
                    s = str(v)
                    if '-' in s:
                        # Dash-separated multi-label: "1-2" → label 1 and 2
                        for part in s.split('-'):
                            vi = int(part.strip())
                            if 0 <= vi < self.num_labels:
                                label[i, vi] = 1.0
                    else:
                        # Single integer label: backward compatible
                        vi = int(float(s))
                        if 0 <= vi < self.num_labels:
                            label[i, vi] = 1.0

        pad_row = np.full((1, self.num_labels), -100, dtype=np.float32)
        label = np.concatenate([pad_row, label, pad_row], axis=0)

        return name, prolabel, label, sequence

    def __len__(self):
        return len(self.names)


def compute_pos_weights(dataset_obj, num_labels):
    """Compute pos_weight for BCEWithLogitsLoss: neg_count / pos_count per label."""
    pos_counts = np.zeros(num_labels, dtype=np.float64)
    total_count = 0
    for i in range(len(dataset_obj)):
        _, _, label, _ = dataset_obj[i]
        valid = label[label[:, 0] != -100]
        pos_counts += valid.sum(axis=0)
        total_count += valid.shape[0]

    neg_counts = total_count - pos_counts
    weights = neg_counts / np.maximum(pos_counts, 1)
    print(f'Token positive counts per label: {pos_counts.tolist()}')
    print(f'Total valid tokens: {total_count}')
    print(f'Auto pos_weight: {weights.tolist()}')
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_parameters():
    parser = argparse.ArgumentParser(description='Multi-LABEL token classifier (DDP)')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t30_150M_UR50D')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of possible labels per token')
    parser.add_argument('--ft_mode', type=str, default='full',
                        help='Fine-tune mode: ["full", "lora"]')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    # Loss args
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['bce', 'bce_weighted', 'focal'],
                        help='Loss: bce / bce_weighted / focal (default: focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--auto_pos_weight', action='store_true',
                        help='Auto-compute pos_weight (neg/pos ratio) per label')
    parser.add_argument('--pos_weight', type=str, default='',
                        help='Manual pos_weight, comma-separated, e.g. "1.0,50.0,3.0"')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Sigmoid threshold for prediction (default: 0.5)')
    # DDP & performance args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--deepspeed', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Model (standard EsmForTokenClassification, no subclass)
# ---------------------------------------------------------------------------
def model_load(model_name, num_labels, dropout, ft_mode, lora_rank):
    model = EsmForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels,
        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)

    if ft_mode == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False, bias="none", use_rslora=True,
            r=lora_rank, lora_alpha=16 * (lora_rank ** .5), lora_dropout=0.05,
            target_modules=["query", "key", "value",
                            "EsmSelfOutput.dense", "EsmIntermediate.dense",
                            "EsmOutput.dense", "EsmContactPredictionHead.regression"])
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def train_args_prepare(args):
    mod_name = args.model_name.split('/')[-1]
    folder_name = (f"finetune/{mod_name}-{args.ft_mode}-Token-multilabel-{args.num_labels}"
                   f"-{args.loss_type}-gamma{args.focal_gamma}-ddp")
    if args.ft_mode == 'lora':
        folder_name = (f"finetune/{mod_name}-lora-r{args.lora_rank}-Token-multilabel-{args.num_labels}"
                       f"-{args.loss_type}-gamma{args.focal_gamma}-ddp")
    return TrainingArguments(
        output_dir=folder_name,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        fp16=args.fp16, bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        push_to_hub=False, deepspeed=args.deepspeed,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
THRESHOLD = 0.5

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    num_labels = logits.shape[-1]

    logits_flat = logits.reshape(-1, num_labels)
    labels_flat = labels.reshape(-1, num_labels)
    mask = (labels_flat[:, 0] != -100)
    logits_valid = logits_flat[mask]
    labels_valid = labels_flat[mask].astype(int)

    probs = 1 / (1 + np.exp(-logits_valid))
    preds = (probs > THRESHOLD).astype(int)

    result = {
        "f1_micro": f1_score(labels_valid, preds, average='micro', zero_division=0),
        "f1_macro": f1_score(labels_valid, preds, average='macro', zero_division=0),
        "f1_samples": f1_score(labels_valid, preds, average='samples', zero_division=0),
        "precision_micro": precision_score(labels_valid, preds, average='micro', zero_division=0),
        "recall_micro": recall_score(labels_valid, preds, average='micro', zero_division=0),
        "exact_match": float((preds == labels_valid).all(axis=1).mean()),
    }
    for c in range(num_labels):
        result[f"f1_label{c}"] = f1_score(labels_valid[:, c], preds[:, c], zero_division=0)
        result[f"recall_label{c}"] = recall_score(labels_valid[:, c], preds[:, c], zero_division=0)
        result[f"precision_label{c}"] = precision_score(labels_valid[:, c], preds[:, c], zero_division=0)

    return result


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def dataset_prepare(dataset_obj, tokenizer, num_labels):
    labels_list = [dataset_obj[i][2] for i in range(len(dataset_obj))]
    sequences = dataset_obj.sequences
    tokenized = tokenizer(sequences)
    ds = datasets.Dataset.from_dict(tokenized)
    ds = ds.add_column("labels", [lab.tolist() for lab in labels_list])
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global THRESHOLD
    args = get_parameters()
    THRESHOLD = args.threshold

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f'Available GPUs: {torch.cuda.device_count()}')
    effective_bs = args.batch_size * max(1, torch.cuda.device_count()) * args.gradient_accumulation_steps
    print(f'Effective total batch size: {effective_bs}')
    print(f'Loss type: {args.loss_type}, focal_gamma: {args.focal_gamma}')
    print(f'Num labels: {args.num_labels} (multi-label mode)')

    print('Loading data...')
    train_set = MyDataset(os.path.join(args.data_path, 'train.csv'), args.label_path, args.num_labels)

    # Determine pos_weight
    pos_weight = None
    if args.pos_weight:
        pos_weight = torch.tensor([float(w) for w in args.pos_weight.split(',')], dtype=torch.float32)
        print(f'Manual pos_weight: {pos_weight.tolist()}')
    elif args.auto_pos_weight:
        pos_weight = compute_pos_weights(train_set, args.num_labels)

    print('Loading model...')
    model, tokenizer = model_load(args.model_name, args.num_labels, args.dropout,
                                  args.ft_mode, args.lora_rank)

    data_collator = MultiLabelDataCollator(tokenizer, args.num_labels)
    train_dataset = dataset_prepare(train_set, tokenizer, args.num_labels)

    val_set = MyDataset(os.path.join(args.data_path, 'val.csv'), args.label_path, args.num_labels)
    val_dataset = dataset_prepare(val_set, tokenizer, args.num_labels)

    train_args = train_args_prepare(args)

    trainer = MultilabelTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        loss_type=args.loss_type,
        pos_weight=pos_weight,
        focal_gamma=args.focal_gamma,
        num_labels=args.num_labels,
    )

    print('Begin training...')
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(os.path.join(train_args.output_dir, 'best_model'))


if __name__ == '__main__':
    main()
