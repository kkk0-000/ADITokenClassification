"""
Multi-class token classification with LoRA + multi-GPU DDP + Focal Loss / class-weighted loss.

Launch examples:
  # 4 GPUs + Focal Loss + LoRA:
  accelerate launch --num_processes 4 train_Tokens_LoRA_multiclass_ddp.py \
      --num_labels 3 --loss_type focal --focal_gamma 2.0 --auto_class_weight --ft_mode lora ...

  # torchrun:
  torchrun --nproc_per_node=4 train_Tokens_LoRA_multiclass_ddp.py --num_labels 3 --loss_type focal ...

  # Single GPU fallback:
  python train_Tokens_LoRA_multiclass_ddp.py --num_labels 3 ...
"""

import os, sys, re
import argparse
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import torch
import datasets
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union

from torch.utils.data import Dataset
from transformers import EsmModel, EsmForTokenClassification
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig, TaskType
from focal_loss import FocalLoss

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'


# ---------------------------------------------------------------------------
# Custom Trainer: only override compute_loss, keeps model untouched
# ---------------------------------------------------------------------------
class FocalTrainer(Trainer):
    """Trainer with pluggable loss: CE / weighted-CE / Focal Loss."""

    def __init__(self, *args, loss_type='ce', class_weights=None,
                 focal_gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        device = logits.device
        num_labels = logits.shape[-1]
        weights = self.class_weights.to(device) if self.class_weights is not None else None

        if self.loss_type == 'focal':
            loss_fct = FocalLoss(alpha=weights, gamma=self.focal_gamma,
                                 reduction='mean', ignore_index=-100)
        elif self.loss_type == 'ce_weighted':
            loss_fct = CrossEntropyLoss(weight=weights,
                                        label_smoothing=self.label_smoothing)
        else:
            loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)

        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Custom model class (same as original train_Tokens_LoRA.py, for label_smoothing CE)
# Only needed if ft_mode == "freeze"; otherwise use EsmForTokenClassification directly
# ---------------------------------------------------------------------------
@dataclass
class TokenClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MyTokensClassification(EsmForTokenClassification):
    """Same as original LoRA version — only adds freeze support."""

    def __init__(self, config, freeze=False):
        super().__init__(config)
        self.num_labels = config.num_labels
        if freeze:
            for param in self.esm.parameters():
                param.requires_grad = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data_table, label_path):
        df = pd.read_csv(data_table)
        self.names = df['ProId'].tolist()
        self.sequences = df['Sequence'].tolist()
        self.prolabels = df['Class'].tolist()
        self.label_path = label_path

    def __getitem__(self, index):
        name = self.names[index]
        sequence = self.sequences[index]
        prolabel = torch.tensor(self.prolabels[index])
        label = torch.from_numpy(np.pad(np.array([0] * len(sequence)),
                                        (1, 1), mode='constant', constant_values=-100))
        if prolabel >= 1:
            label = torch.from_numpy(np.pad(
                np.load(os.path.join(self.label_path, name + '.npy')).astype(np.int64),
                (1, 1), mode='constant', constant_values=-100))
        return name, prolabel, label, sequence

    def __len__(self):
        return len(self.names)


def compute_class_weights_from_labels(dataset_obj, num_classes):
    """Scan all token labels to compute inverse-frequency class weights."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for i in range(len(dataset_obj)):
        _, _, label, _ = dataset_obj[i]
        valid = label[label != -100]
        for c in range(num_classes):
            counts[c] += (valid == c).sum().item()

    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    print(f'Token class counts: {counts.tolist()}')
    print(f'Auto class weights: {weights.tolist()}')
    return weights.float()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def get_parameters():
    parser = argparse.ArgumentParser(description='Multi-class token classifier LoRA (DDP + Focal)')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-device batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--ft_mode', type=str, default='full',
                        help='Fine-tune mode: ["full", "lora", "freeze"]')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    # Loss args
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['ce', 'ce_weighted', 'focal'],
                        help='Loss type: ce / ce_weighted / focal (default: focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--auto_class_weight', action='store_true',
                        help='Auto-compute inverse-frequency class weights from training data')
    parser.add_argument('--class_weights', type=str, default='',
                        help='Manual class weights, comma-separated, e.g. "1.0,100.0,5.0"')
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


target_modules = []
for i in range(33):
    target_modules.append("encoder.layer.%d.output.dense" % i)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def model_load(model_name, num_labels, ft_mode, lora_rank):
    print("model_name", model_name)
    if ft_mode == "freeze":
        model = MyTokensClassification.from_pretrained(model_name, num_labels=num_labels, freeze=True)
    else:
        model = EsmForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    if ft_mode == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            bias="none",
            use_rslora=True,
            r=lora_rank,
            lora_alpha=16 * (lora_rank ** .5),
            lora_dropout=0.05,
            target_modules=target_modules,)
        model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return model, tokenizer, data_collator


def train_args_prepare(args):
    mol_name = args.model_name.split('/')[-1]

    if args.ft_mode == 'lora':
        folder_name = (f"finetune/{mol_name}-lora-r{args.lora_rank}-TokenCLS-multiclass-{args.num_labels}"
                       f"-{args.loss_type}-gamma{args.focal_gamma}-ddp")
    elif args.ft_mode == 'freeze':
        folder_name = (f"finetune/{mol_name}-freeze-TokenCLS-multiclass-{args.num_labels}"
                       f"-{args.loss_type}-gamma{args.focal_gamma}-ddp")
    else:
        folder_name = (f"finetune/{mol_name}-full-TokenCLS-multiclass-{args.num_labels}"
                       f"-{args.loss_type}-gamma{args.focal_gamma}-ddp")

    train_args = TrainingArguments(
        output_dir=folder_name,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_total_limit=args.save_total_limit,
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        push_to_hub=False,
        deepspeed=args.deepspeed,
    )
    return train_args


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    num_classes = logits.shape[-1]
    labels = labels.reshape((-1))
    probs = torch.softmax(torch.tensor(logits.reshape((-1, num_classes))), dim=1).numpy()
    preds = np.argmax(logits, axis=2).reshape((-1,))
    mask = labels != -100
    preds = preds[mask]
    probs = probs[mask]
    labels = labels[mask]

    result = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average='macro', zero_division=0),
        "precision_macro": precision_score(labels, preds, average='macro', zero_division=0),
        "recall_macro": recall_score(labels, preds, average='macro', zero_division=0),
        "f1_weighted": f1_score(labels, preds, average='weighted', zero_division=0),
        "precision_weighted": precision_score(labels, preds, average='weighted', zero_division=0),
        "recall_weighted": recall_score(labels, preds, average='weighted', zero_division=0),
    }
    for c in range(num_classes):
        binary_preds = (preds == c).astype(int)
        binary_labels = (labels == c).astype(int)
        result[f"f1_class{c}"] = f1_score(binary_labels, binary_preds, zero_division=0)
        result[f"recall_class{c}"] = recall_score(binary_labels, binary_preds, zero_division=0)

    try:
        if num_classes == 2:
            result["roc_auc"] = roc_auc_score(labels, probs[:, 1])
        else:
            result["roc_auc"] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        result["roc_auc"] = 0.0
    return result


def dataset_prepare(MyDataset_obj, tokenizer):
    token_labels = [MyDataset_obj[i][2].numpy() for i in range(len(MyDataset_obj))]
    sequences = MyDataset_obj.sequences
    tokenized = tokenizer(sequences)
    dataset = datasets.Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", token_labels)
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_parameters()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f'Available GPUs: {torch.cuda.device_count()}')
    print(f'Per-device batch size: {args.batch_size}')
    print(f'Gradient accumulation steps: {args.gradient_accumulation_steps}')
    effective_bs = args.batch_size * max(1, torch.cuda.device_count()) * args.gradient_accumulation_steps
    print(f'Effective total batch size: {effective_bs}')
    print(f'Loss type: {args.loss_type}, focal_gamma: {args.focal_gamma}')

    print('Loading data...')
    train_set = MyDataset(os.path.join(args.data_path, 'train.csv'), args.label_path)

    # Determine class weights
    class_weights = None
    if args.class_weights:
        class_weights = torch.tensor([float(w) for w in args.class_weights.split(',')],
                                     dtype=torch.float32)
        print(f'Manual class weights: {class_weights.tolist()}')
    elif args.auto_class_weight:
        class_weights = compute_class_weights_from_labels(train_set, args.num_labels)

    print('Loading model...')
    model, tokenizer, data_collator = model_load(
        args.model_name, args.num_labels, args.ft_mode, args.lora_rank)

    train_dataset = dataset_prepare(train_set, tokenizer)

    val_set = MyDataset(os.path.join(args.data_path, 'val.csv'), args.label_path)
    val_dataset = dataset_prepare(val_set, tokenizer)

    train_args = train_args_prepare(args)

    # Use FocalTrainer — loss logic lives in Trainer, not in model
    trainer = FocalTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        loss_type=args.loss_type,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    print('Begin training...')
    trainer.train()

    if trainer.is_world_process_zero():
        print('Saving best model...')
        trainer.save_model(os.path.join(train_args.output_dir, 'best_model'))


if __name__ == '__main__':
    main()
