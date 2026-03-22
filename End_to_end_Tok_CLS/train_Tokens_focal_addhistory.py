# !module load cuda/12.1
# !export LD_LIBRARY_PATH=/YOUR_HOME/miniconda3/envs/gnn/lib:$LD_LIBRARY_PATH

import os,sys,re
import copy
import math
from typing import List, Optional, Tuple, Union
import argparse
import random
import time
import evaluate
import numpy as np
import pandas as pd
import torch
import datasets
import torch.nn as nn
from torch.nn import Dropout, Linear, CrossEntropyLoss
import torch
import json

import torch.utils.checkpoint
from torch.utils.data import Dataset
from transformers import EsmModel, EsmPreTrainedModel,EsmForTokenClassification
from transformers.models.esm.modeling_esm import EsmClassificationHead
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import TrainerCallback
from sklearn.metrics import matthews_corrcoef

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

## Models
@dataclass
class SequenceTokenClassifierOutput(ModelOutput):

    loss_seq: Optional[torch.FloatTensor] = None
    loss_tok: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits_seq: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

# DataSet preparing
class MyDataset(Dataset):

    def __init__(self, data_table, label_path):
        df = pd.read_csv(data_table)
        self.names = df['ProId'].tolist()
        self.sequences = df['Sequence'].tolist()
        self.prolabels = df['Class'].tolist() ## label for Sequence
        self.label_path = label_path ## label for Token

    def __getitem__(self, index):
        name = self.names[index]
        sequence = self.sequences[index]
        prolabel = torch.tensor(self.prolabels[index])
        label = torch.from_numpy(np.pad(np.array([0]*len(sequence)),
                                        (1,1), mode='constant', constant_values=-100))
        if prolabel == 1:
            label = torch.from_numpy(np.pad(np.load(os.path.join(self.label_path,name+'.npy')),
                                            (1,1), mode='constant', constant_values=-100))        
        return name, prolabel, label, sequence

    def __len__(self):
        return len(self.names)
    
    def get_num_samples_per_class(self):
        return torch.bincount(torch.tensor(self.prolabels)).long().tolist()

def get_class_weight(num_samples_per_class):
    normalized_weights = [ 1/(count/sum(num_samples_per_class)) for count in num_samples_per_class]
    print(normalized_weights)
    return torch.FloatTensor(normalized_weights) 

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='AMP token classifer')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--model_name', type=str, default='../../basemodels/esm2_t30_150M_UR50D' , help="['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D']")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--ft_mode', type=str, default='full',help='Fine-tune mode: ["full", "lora"]')
    ## lora 参数
    parser.add_argument('--lora_rank', type=int, default=8,help='LoRa rank. (default 8).')
    parser.add_argument('--lora_alpha', type=int, default=16,help='LoRa alpha. (default 16).')
    parser.add_argument('--lora_dropout', type=float, default=0.1,help='LoRa drouput. (default 0.1).')

    ## loss 函数
    parser.add_argument('--loss_function', type=str, default='ce',choices=['ce', 'focal', 'pu'],
                        help='Loss function: ce (cross-entropy), focal (focal loss), pu (nnPU loss)')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of loss for token-classifiction. (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--outdir', type=str, default='finetune', help='outdir path')
    ## 并行版本
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                    help='num workers for dataloader')
    parser.add_argument('--fp16', type=lambda x: str(x).lower() == 'true', default=True,
                        help='use fp16 mixed precision')
    parser.add_argument('--bf16', type=lambda x: str(x).lower() == 'true', default=False,
                        help='use bf16 mixed precision')
    parser.add_argument('--ddp_find_unused_parameters', type=lambda x: str(x).lower() == 'true', default=False,
                        help='DDP find_unused_parameters')
    # parser.add_argument('--cuda', type=int, default=0, help='CUDA number (default: 0). ["None",0,1,2,3].')
    # parser.add_argument('--alpha', type=float, default=1.0, help='Weight for CE loss (default 1.0)')
    # parser.add_argument('--beta', type=float, default=1.0, help='Weight for focal loss (default 1.0)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal parameter (default: 2.0)')
    parser.add_argument('--focal_alpha', type=float, default=2.0, help='focal parameter (default: 2.0)')

    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of loss for sequence-classifiction. (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=1.0, help='Weight for focal loss (default 1.0)')
    # Training arguments
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    args = parser.parse_args()
    return args

## Model preparing
def model_load(model_name,num_labels,dropout,ft_mode,lora_rank=8,lora_alpha=16,lora_dropout=0.1):
    #token_dropout = True if dropout > 0.0 else False
    # model = EsmForTokenClassification.from_pretrained(model_name, num_labels=num_labels, 
    #                                                   hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
    #                                                   #token_dropout=token_dropout
    #                                                   )
    model = EsmForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        local_files_only=True
    )
    if ft_mode == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            bias="none",
            use_rslora = True,
            r=lora_rank,
            # lora_alpha=16*(lora_rank**.5),
            lora_alpha = lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "query",
                "key",
                "value",
                "EsmSelfOutput.dense",
                "EsmIntermediate.dense",
                "EsmOutput.dense",
                "EsmContactPredictionHead.regression",
                "classifier_seq.dense",
                "classifier_seq.out_proj",
                "classifier_tok",
            ])
        model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return model, tokenizer, data_collator
    
def train_args_prepare(model_name, learning_rate, batch_size, num_train_epochs, weight_decay,
                       ft_mode, lora_rank, total_steps=None, warmup_ratio=0.1, outdir=None,
                      dataloader_num_workers=4,
                       fp16=True,
                       bf16=False,
                       ddp_find_unused_parameters=False,gradient_accumulation_steps=1):
    mod_name = model_name.split('/')[-1]
    folder_name = f"{outdir}/{mod_name}-full-ft-for-Token-classification"
    if ft_mode == 'lora':
        folder_name = f"{outdir}/{mod_name}-{ft_mode}-rank-{lora_rank}-ft-for-Sequence-and-Token-classification"

    warmup_steps = int(total_steps * warmup_ratio) if total_steps is not None else 0

    train_args = TrainingArguments(
        output_dir=folder_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",   # 关键：显式 eval_f1
        greater_is_better=True,
        save_total_limit=None,
        push_to_hub=False,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        fp16=fp16,
        bf16=bf16,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        remove_unused_columns=False,        # 自定义compute_loss建议开
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    return train_args

# 用于保存历史的变量
train_history = []
eval_history = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape((-1))
    probs = torch.softmax(torch.tensor(logits.reshape((-1, logits.shape[-1]))), dim=1).numpy()
    preds = np.argmax(logits, axis=2).reshape((-1,))
    mask = labels != -100
    preds = preds[mask]
    probs = probs[mask]
    labels = labels[mask]

    # 计算指标
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    pre = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except:
        roc_auc = 0.0
    
    # 混淆矩阵等
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "auc": roc_auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "mcc": matthews_corrcoef(labels, preds),
        "specificity": specificity,
    }

    # 这里可以把val_metrics加入eval_history，稍后写入文件
    # 实际要做是多线程安全、文件锁等，这里仅示意
    if trainer_state := globals().get('trainer_state'):
        epoch = trainer_state.epoch if hasattr(trainer_state, 'epoch') else None
    else:
        epoch = None
    
    # if epoch is not None:
    #     eval_history.append({"epoch": epoch, "val_metrics": metrics})
    # else:
    #     # 也可能外部调用时没epoch
    #     pass

    return metrics

# class HistoryCallback(TrainerCallback):
#     def __init__(self, output_dir: str, train_dataset, filename: str = "train_history.json"):
#         self.output_dir = output_dir
#         self.filename = filename
#         self.train_dataset = train_dataset
#         self.epoch_records = {}  # {epoch: {...}}

#     def _save(self):
#         os.makedirs(self.output_dir, exist_ok=True)
#         fp = os.path.join(self.output_dir, self.filename)
#         data = [self.epoch_records[k] for k in sorted(self.epoch_records.keys())]
#         with open(fp, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#     def _get_last_train_loss_from_log_history(self, state, epoch):
#         # 在 log_history 里找该 epoch 最近一条 train loss
#         last_loss = None
#         for row in state.log_history:
#             if "epoch" in row and "loss" in row:
#                 ep = float(row["epoch"])
#                 if abs(ep - float(epoch)) < 1e-8:
#                     last_loss = row["loss"]
#         return last_loss

#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         if not state.is_world_process_zero:
#             return

#         trainer = kwargs.get("model")  # 这里拿不到 trainer 本体，下面从 kwargs 取
#         # HF 在 callback kwargs 里通常会给 trainer
#         trainer_obj = kwargs.get("trainer", None)
#         if trainer_obj is None:
#             # 兼容某些版本：尝试从 kwargs 里取
#             for v in kwargs.values():
#                 if isinstance(v, Trainer):
#                     trainer_obj = v
#                     break
#         if trainer_obj is None:
#             return

#         if metrics is None:
#             metrics = {}

#         cur_epoch = state.epoch
#         if cur_epoch is None:
#             return
#         ep = int(cur_epoch) if float(cur_epoch).is_integer() else float(cur_epoch)

#         # val
#         val_loss = metrics.get("eval_loss")
#         val_metrics = extract_core_metrics(metrics, "eval")

#         # train loss（从日志找）
#         train_loss = self._get_last_train_loss_from_log_history(state, ep)

#         # train metrics（关键：每个 epoch 额外跑一次 train predict）
#         train_output = trainer_obj.predict(self.train_dataset, metric_key_prefix="train")
#         train_metrics = extract_core_metrics(train_output.metrics, "train")

#         self.epoch_records[ep] = {
#             "epoch": ep,
#             "train_loss": train_loss,
#             "train_metrics": train_metrics,
#             "val_loss": val_loss,
#             "val_metrics": val_metrics
#         }

#         self._save()

#     def on_train_end(self, args, state, control, **kwargs):
#         if state.is_world_process_zero:
#             self._save()



## Data preparing
def dataset_prepare(MyDataset_obj, tokenizer):
    token_labels = [ MyDataset_obj[i][2].numpy() for i in range(len(MyDataset_obj)) ]
    sequences = MyDataset_obj.sequences
    tokenized = tokenizer(sequences)
    dataset = datasets.Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", token_labels)
    return dataset


# FocalLoss实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        logits_ = logits.view(-1, logits.size(-1))
        targets_ = targets.view(-1)

        ce_loss = self.ce_loss(logits_, targets_)  # [N]

        probs = torch.exp(-ce_loss)  # p_t
        focal_loss = self.alpha * (1 - probs) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss        

class TrainerWithMixedLoss(Trainer):
    def __init__(
        self,
        *args,
        loss_alpha=1.0,
        loss_beta=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        loss_function='ce',
        history_recorder=None,
        train_dataset_for_metrics=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_function = loss_function

        # 新增：每个epoch记录器
        self.history_recorder = history_recorder
        self.train_dataset_for_metrics = train_dataset_for_metrics
        self._recorded_eval_steps = set()  # 防止同一步重复记录

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = logits.size(-1)   # <- 新增
    
        if self.loss_function == 'ce':
            criterion = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            total_loss = ce_loss
    
        elif self.loss_function == 'focal':
            criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, ignore_index=-100)
            focal_loss = criterion(logits.view(-1, num_labels), labels.view(-1))
            total_loss = focal_loss
    
        else:  # mixed
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = ce_criterion(logits.view(-1, num_labels), labels.view(-1))
    
            focal_criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, ignore_index=-100)
            focal_loss = focal_criterion(logits.view(-1, num_labels), labels.view(-1))
    
            total_loss = self.loss_alpha * ce_loss + self.loss_beta * focal_loss
    
        return (total_loss, outputs) if return_outputs else total_loss


    def _find_train_loss_for_epoch(self, epoch_int):
        # 从log_history里抓该epoch最后一个训练loss
        last_loss = None
        for row in self.state.log_history:
            if "epoch" not in row:
                continue
            ep = float(row["epoch"])
            if ep.is_integer() and int(ep) == epoch_int:
                if "loss" in row and not any(k.startswith("eval_") for k in row.keys()):
                    last_loss = row["loss"]
        return last_loss


def extract_core_metrics(metrics_dict, prefix):
    return {
        "accuracy": metrics_dict.get(f"{prefix}_accuracy"),
        "precision": metrics_dict.get(f"{prefix}_precision"),
        "recall": metrics_dict.get(f"{prefix}_recall"),
        "f1": metrics_dict.get(f"{prefix}_f1"),
        "auc": metrics_dict.get(f"{prefix}_auc"),
        "mcc": metrics_dict.get(f"{prefix}_mcc"),
        "specificity": metrics_dict.get(f"{prefix}_specificity"),
        "tp": metrics_dict.get(f"{prefix}_tp"),
        "tn": metrics_dict.get(f"{prefix}_tn"),
        "fp": metrics_dict.get(f"{prefix}_fp"),
        "fn": metrics_dict.get(f"{prefix}_fn"),
    }
def build_history_from_log(log_history):
        epoch_map = {}
        for row in log_history:
            if "epoch" not in row:
                continue
            epf = float(row["epoch"])
            if not epf.is_integer():
                continue
            ep = int(epf)
    
            if ep not in epoch_map:
                epoch_map[ep] = {"epoch": ep}
    
            # train row
            if "loss" in row and not any(k.startswith("eval_") for k in row.keys()):
                epoch_map[ep]["train_loss"] = row["loss"]
    
            # eval row
            if "eval_loss" in row:
                epoch_map[ep]["val_loss"] = row.get("eval_loss")
                epoch_map[ep]["val_metrics"] = {
                    "accuracy": row.get("eval_accuracy"),
                    "precision": row.get("eval_precision"),
                    "recall": row.get("eval_recall"),
                    "f1": row.get("eval_f1"),
                    "auc": row.get("eval_auc"),
                    "mcc": row.get("eval_mcc"),
                    "specificity": row.get("eval_specificity"),
                    "tp": row.get("eval_tp"),
                    "tn": row.get("eval_tn"),
                    "fp": row.get("eval_fp"),
                    "fn": row.get("eval_fn"),
                }
    
        return [epoch_map[k] for k in epoch_map.keys()]
def trainer_prepare(model, train_args, train_dataset, test_dataset, tokenizer, data_collator, compute_metrics,
                    alpha=1.0, beta=1.0, focal_alpha=0.25, focal_gamma=2.0,
                    loss_function='ce', outdir=None):
    trainer = TrainerWithMixedLoss(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        loss_alpha=alpha,
        loss_beta=beta,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        loss_function=loss_function,
    )
    return trainer


# def trainer_prepare(model, train_args, train_dataset, test_dataset, tokenizer, data_collator, compute_metrics,
#                     alpha=1.0, beta=1.0, focal_alpha=0.25, focal_gamma=2.0,
#                     loss_function='ce', outdir=None):
#     recorder = EpochHistoryRecorder(output_dir=outdir, filename="train_history.json")

#     trainer = TrainerWithMixedLoss(
#         model=model,
#         args=train_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,   # 你这里传的是val_dataset
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#         data_collator=data_collator,
#         loss_alpha=alpha,
#         loss_beta=beta,
#         focal_alpha=focal_alpha,
#         focal_gamma=focal_gamma,
#         loss_function=loss_function,
#         history_recorder=recorder,
#         train_dataset_for_metrics=train_dataset
#     )
#     return trainer

# class EpochMetricsRecorderCallback(TrainerCallback):
#     """
#     每次 eval（按你设置是每个 epoch）后：
#     - 从 log_history 取该 epoch 的 train_loss
#     - 用当前epoch权重在 train_dataset 上 predict -> train_metrics
#     - 使用本次 evaluate 的 metrics -> val_loss/val_metrics
#     - 持续写入 train_history.json
#     """
#     def __init__(self, train_dataset, output_dir, filename="train_history.json"):
#         self.train_dataset = train_dataset
#         self.output_dir = output_dir
#         self.filename = filename
#         self.records = {}  # epoch -> record

#     def _save(self):
#         os.makedirs(self.output_dir, exist_ok=True)
#         path = os.path.join(self.output_dir, self.filename)
#         data = [self.records[k] for k in sorted(self.records.keys())]
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#     def _find_train_loss(self, log_history, ep):
#         last_loss = None
#         for row in log_history:
#             if "epoch" not in row:
#                 continue
#             ep_row = float(row["epoch"])
#             if ep_row.is_integer() and int(ep_row) == ep:
#                 # 训练日志通常有 loss 且不含 eval_*
#                 if "loss" in row and not any(k.startswith("eval_") for k in row.keys()):
#                     last_loss = row["loss"]
#         return last_loss

#     def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
#         if not state.is_world_process_zero:
#             return
#         if state.epoch is None:
#             return

#         trainer = kwargs.get("trainer", None)
#         if trainer is None:
#             # 某些版本不会传 trainer，这里直接跳过，避免写空
#             return

#         epf = float(state.epoch)
#         if not epf.is_integer():
#             return
#         ep = int(epf)

#         # 当前epoch的train_loss（从日志中取）
#         train_loss = self._find_train_loss(state.log_history, ep)

#         # 当前epoch模型在train集评估（重点）
#         train_out = trainer.predict(self.train_dataset, metric_key_prefix="train")
#         train_metrics = extract_core_metrics(train_out.metrics, "train")

#         # 本次evaluate得到的val指标
#         metrics = metrics or {}
#         val_loss = metrics.get("eval_loss")
#         val_metrics = extract_core_metrics(metrics, "eval")

#         self.records[ep] = {
#             "epoch": ep,
#             "train_loss": train_loss,
#             "train_metrics": train_metrics,
#             "val_loss": val_loss,
#             "val_metrics": val_metrics
#         }

#         self._save()

#     def on_train_end(self, args, state, control, **kwargs):
#         if state.is_world_process_zero:
#             self._save()
import os
import json

class EpochHistoryCallback(TrainerCallback):
    def __init__(self, output_dir, total_epochs, filename="train_history.json"):
        self.output_dir = output_dir
        self.total_epochs = int(total_epochs)
        self.filename = filename

    def _path(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.filename)

    def _extract_from_log_history(self, log_history):
        epoch_map = {}

        for row in log_history:
            if "epoch" not in row:
                continue
            epf = float(row["epoch"])
            if not epf.is_integer():
                continue
            ep = int(epf)

            if ep not in epoch_map:
                epoch_map[ep] = {
                    "epoch": ep,
                    "train_loss": None,
                    "train_metrics": None,
                    "val_loss": None,
                    "val_metrics": None
                }

            # train
            if "loss" in row and not any(k.startswith("eval_") for k in row.keys()):
                epoch_map[ep]["train_loss"] = row.get("loss")

            # eval
            if "eval_loss" in row:
                epoch_map[ep]["val_loss"] = row.get("eval_loss")
                epoch_map[ep]["val_metrics"] = {
                    "accuracy": row.get("eval_accuracy"),
                    "precision": row.get("eval_precision"),
                    "recall": row.get("eval_recall"),
                    "f1": row.get("eval_f1"),
                    "auc": row.get("eval_auc"),
                    "mcc": row.get("eval_mcc"),
                    "specificity": row.get("eval_specificity"),
                    "tp": row.get("eval_tp"),
                    "tn": row.get("eval_tn"),
                    "fp": row.get("eval_fp"),
                    "fn": row.get("eval_fn"),
                }

        # 补全 1..total_epochs
        data = []
        for ep in range(1, self.total_epochs + 1):
            data.append(epoch_map.get(ep, {
                "epoch": ep,
                "train_loss": None,
                "train_metrics": None,
                "val_loss": None,
                "val_metrics": None
            }))
        return data

    def _save(self, state):
        data = self._extract_from_log_history(state.log_history)
        with open(self._path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._save(state)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._save(state)


# class EpochHistoryRecorder:
#     def __init__(self, output_dir, filename="train_history.json"):
#         self.output_dir = output_dir
#         self.filename = filename
#         self.records = {}  # key: global_step -> record dict

#     def _path(self):
#         os.makedirs(self.output_dir, exist_ok=True)
#         return os.path.join(self.output_dir, self.filename)

#     def save(self):
#         # 按step排序写出
#         data = [self.records[k] for k in sorted(self.records.keys())]
#         with open(self._path(), "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#     def set_step_record(self, step, record):
#         self.records[int(step)] = record
#         self.save()


## main
def main():
    args = get_parameters()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Device
    # if args.cuda is None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # else:
    #     if torch.cuda.is_available():
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    print('Loading model...')
    model, tokenizer, data_collator = model_load(
        args.model_name, args.num_classes, args.dropout, args.ft_mode,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )

    print('Loading data...')
    train_set = MyDataset(os.path.join(args.data_path, 'train.csv'), args.label_path)
    val_set   = MyDataset(os.path.join(args.data_path, 'val.csv'), args.label_path)
    test_set  = MyDataset(os.path.join(args.data_path, 'test.csv'), args.label_path)  # 新增

    train_dataset = dataset_prepare(train_set, tokenizer)
    val_dataset   = dataset_prepare(val_set, tokenizer)
    test_dataset  = dataset_prepare(test_set, tokenizer)

    train_batch_size = args.batch_size
    grad_acc = args.gradient_accumulation_steps
    num_epochs = args.epochs
    num_training_steps = (len(train_dataset) // train_batch_size // grad_acc) * num_epochs
    train_args = train_args_prepare(
        args.model_name, args.lr, args.batch_size, args.epochs, args.weight_decay,
        args.ft_mode, args.lora_rank, total_steps=num_training_steps,
        warmup_ratio=args.warmup_ratio, outdir=args.outdir,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        bf16=args.bf16,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # train_args = train_args_prepare(
    #     args.model_name, args.lr, args.batch_size, args.epochs, args.weight_decay,
    #     args.ft_mode, args.lora_rank, total_steps=num_training_steps,
    #     warmup_ratio=args.warmup_ratio, outdir=args.outdir,
    #      dataloader_num_workers=args.dataloader_num_workers,
    #     fp16=args.fp16,
    #     bf16=args.bf16,
    #     ddp_find_unused_parameters=args.ddp_find_unused_parameters
    # )

    trainer = trainer_prepare(
        model, train_args,
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        alpha=args.alpha,
        beta=args.beta,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        loss_function=args.loss_function,
        outdir=train_args.output_dir
    )
    cb = EpochHistoryCallback(
        output_dir=train_args.output_dir,
        total_epochs=int(train_args.num_train_epochs)
    )
    trainer.add_callback(cb)


    print('Begin training...')
    trainer.train() 
    # 保存 best_model
    best_dir = os.path.join(train_args.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    
    # test_results.json
    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    tm = test_output.metrics
    test_results = extract_core_metrics(tm, "test")
    with open(os.path.join(train_args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # training_args.json
    training_cfg = vars(args).copy()
    training_cfg["output_dir"] = train_args.output_dir
    with open(os.path.join(train_args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(training_cfg, f, indent=2, ensure_ascii=False)
    
    # # train_history.json
    # history = build_history_from_log(trainer.state.log_history)
    
    # # 这里给每个epoch填充 train_metrics（同一份：最终模型在train上的结果）
    # train_out = trainer.predict(train_dataset, metric_key_prefix="train")
    # train_metrics = extract_core_metrics(train_out.metrics, "train")
    
    # for x in history:
    #     x["train_metrics"] = train_metrics  # 结构满足你要求
    
    # with open(os.path.join(train_args.output_dir, "train_history.json"), "w", encoding="utf-8") as f:
    #     json.dump(history, f, indent=2, ensure_ascii=False)

    

    # # ---- 保存 best_model 到独立目录 ----
    # best_dir = os.path.join(train_args.output_dir, "best_model")
    # os.makedirs(best_dir, exist_ok=True)
    # trainer.save_model(best_dir)
    # tokenizer.save_pretrained(best_dir)

    # # ---- 测试集评估并保存 test_results.json ----
    # test_output = trainer.predict(test_dataset)
    # tm = test_output.metrics
    # test_results = {
    #     "accuracy": tm.get("test_accuracy"),
    #     "precision": tm.get("test_precision"),
    #     "recall": tm.get("test_recall"),
    #     "f1": tm.get("test_f1"),
    #     "auc": tm.get("test_auc"),
    #     "mcc": tm.get("test_mcc"),
    #     "specificity": tm.get("test_specificity"),
    #     "tp": tm.get("test_tp"),
    #     "tn": tm.get("test_tn"),
    #     "fp": tm.get("test_fp"),
    #     "fn": tm.get("test_fn")
    # }
    # with open(os.path.join(train_args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
    #     json.dump(test_results, f, indent=2, ensure_ascii=False)

    # # ---- 保存 training_args.json ----
    # training_cfg = vars(args).copy()
    # training_cfg["output_dir"] = train_args.output_dir
    # with open(os.path.join(train_args.output_dir, "training_args.json"), "w", encoding="utf-8") as f:
    #     json.dump(training_cfg, f, indent=2, ensure_ascii=False)

    # print(f"[Done] outputs saved in: {train_args.output_dir}")

        
if __name__ == '__main__':

    main()
