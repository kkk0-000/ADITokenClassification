# !module load cuda/12.1
# !export LD_LIBRARY_PATH=/YOUR_HOME/miniconda3/envs/gnn/lib:$LD_LIBRARY_PATH

import os,sys,re
import copy
import math
import json
from typing import List, Optional, Tuple, Union
import argparse
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
import pandas as pd
import torch
import datasets
import torch.nn as nn
from torch.nn import Dropout, Linear, CrossEntropyLoss

import torch.utils.checkpoint
from torch.utils.data import Dataset
from transformers import EsmModel, EsmPreTrainedModel,EsmForTokenClassification
from transformers.models.esm.modeling_esm import EsmClassificationHead
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.utils.generic import ModelOutput
from dataclasses import dataclass
from peft import get_peft_model, LoraConfig, TaskType

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
    parser.add_argument('--lora_rank', type=int, default=8,help='LoRa rank. (default 8).')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--focal', type=float, default=2.0, help='focal parameter (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of loss for sequence-classifiction. (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of loss for token-classifiction. (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA number (default: 0). ["None",0,1,2,3].')
    args = parser.parse_args()
    return args

## Model preparing
def model_load(model_name,num_labels,dropout,ft_mode,lora_rank,focal,alpha,beta):
    #token_dropout = True if dropout > 0.0 else False
    model = EsmForTokenClassification.from_pretrained(model_name, num_labels=num_labels, 
                                                      hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                                                      #token_dropout=token_dropout
                                                      )
    if ft_mode == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            bias="none",
            use_rslora = True,
            r=lora_rank,
            lora_alpha=16*(lora_rank**.5),
            lora_dropout=0.05,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return model, tokenizer, data_collator

def train_args_prepare(model_name, learning_rate, batch_size, num_train_epochs, weight_decay, ft_mode, lora_rank):
    mod_name = model_name.split('/')[-1]
    folder_name = f"finetune/{mod_name}-full-ft-for-Token-classification"
    if ft_mode == 'lora':
        folder_name = f"finetune/{mod_name}-{ft_mode}-rank-{lora_rank}-ft-for-Sequence-and-Token-classification"
    train_args = TrainingArguments(
        folder_name,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        push_to_hub=False,)
    return train_args

TRAINING_HISTORY_FILENAME = 'training_history.json'
TRAINING_SUMMARY_FILENAME = 'training_summary.json'
METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc', 'specificity', 'tp', 'tn', 'fp', 'fn']


def safe_roc_auc_score(labels, probs):
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return 0.0


def compute_binary_metrics(labels, preds, probs=None):
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    result = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds) if len(set(labels.tolist())) > 1 else 0.0,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if probs is not None:
        result["roc_auc"] = safe_roc_auc_score(labels, probs)
    else:
        result["roc_auc"] = 0.0
    result["auc"] = result["roc_auc"]
    return result


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape((-1))
    probs = torch.softmax(torch.tensor(logits.reshape((-1, logits.shape[-1]))), dim=1).numpy()
    preds = np.argmax(logits, axis=2).reshape((-1,))
    mask = labels != -100
    preds = preds[mask]
    probs = probs[mask]
    labels = labels[mask]
    if logits.shape[-1] == 2:
        return compute_binary_metrics(labels, preds, probs=probs[:, 1])

    result = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro', zero_division=0),
        "precision": precision_score(labels, preds, average='macro', zero_division=0),
        "recall": recall_score(labels, preds, average='macro', zero_division=0),
        "roc_auc": 0.0,
        "auc": 0.0,
        "mcc": 0.0,
        "specificity": 0.0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
    }
    return result

## Data preparing
def dataset_prepare(MyDataset_obj, tokenizer):
    token_labels = [ MyDataset_obj[i][2].numpy() for i in range(len(MyDataset_obj)) ]
    sequences = MyDataset_obj.sequences
    tokenized = tokenizer(sequences)
    dataset = datasets.Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", token_labels)
    return dataset

## Trainer preparing
def trainer_prepare(model, train_args, train_dataset, test_dataset, tokenizer, data_collator, compute_metrics):
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,)
    return trainer


def to_serializable_float(value):
    if value is None:
        return None
    return float(value)


def to_serializable_epoch(value):
    if value is None:
        return None
    if abs(float(value) - round(float(value))) < 1e-9:
        return int(round(float(value)))
    return float(value)


def simplify_prefixed_metrics(metrics, prefix):
    simplified = {}
    for key in METRIC_KEYS:
        metric_key = f'{prefix}_{key}'
        if metric_key in metrics:
            value = metrics[metric_key]
            simplified[key] = int(value) if key in {'tp', 'tn', 'fp', 'fn'} else to_serializable_float(value)
    return simplified


def find_latest_training_loss(log_history):
    for entry in reversed(log_history):
        if 'loss' in entry:
            return float(entry['loss'])
    return None


def find_latest_epoch(log_history):
    for entry in reversed(log_history):
        if 'epoch' in entry:
            return float(entry['epoch'])
    return None


def save_json_file(output_path, payload):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_training_args(args, output_dir):
    training_args_payload = vars(args).copy()
    training_args_payload['output_dir'] = output_dir
    training_args_path = os.path.join(output_dir, 'training_args.json')
    save_json_file(training_args_path, training_args_payload)
    return training_args_path


def save_training_summary(output_dir, history):
    if not history:
        return None
    best_entry = max(history, key=lambda item: item['val_metrics'].get('f1', 0.0))
    summary = {
        'num_epochs': len(history),
        'best_epoch': best_entry['epoch'],
        'best_val_loss': best_entry['val_loss'],
        'best_val_metrics': best_entry['val_metrics'],
    }
    save_json_file(os.path.join(output_dir, TRAINING_SUMMARY_FILENAME), summary)
    return summary


class StatisticsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_history = []
        self._collecting_train_statistics = False

    def _evaluate_train_split(self, ignore_keys=None):
        self._collecting_train_statistics = True
        try:
            return super().evaluate(
                eval_dataset=self.train_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix='train',
            )
        finally:
            self._collecting_train_statistics = False

    def _record_training_history(self, eval_metrics, train_metrics):
        history_entry = {
            'epoch': to_serializable_epoch(find_latest_epoch(self.state.log_history) or self.state.epoch),
            'train_loss': to_serializable_float(train_metrics.get('train_loss', find_latest_training_loss(self.state.log_history))),
            'val_loss': to_serializable_float(eval_metrics.get('eval_loss')),
            'train_metrics': simplify_prefixed_metrics(train_metrics, 'train'),
            'val_metrics': simplify_prefixed_metrics(eval_metrics, 'eval'),
        }
        self.training_history.append(history_entry)
        save_json_file(os.path.join(self.args.output_dir, TRAINING_HISTORY_FILENAME), self.training_history)
        save_training_summary(self.args.output_dir, self.training_history)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        should_collect_history = (
            metric_key_prefix == 'eval'
            and not self._collecting_train_statistics
            and self.train_dataset is not None
        )
        if should_collect_history:
            train_metrics = self._evaluate_train_split(ignore_keys=ignore_keys)
            self._record_training_history(metrics, train_metrics)
        return metrics

## main
def get_parameters_v2():
    parser = argparse.ArgumentParser(description='AMP token classifer with best-model tracking')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--model_name', type=str, default='../../basemodels/esm2_t30_150M_UR50D', help="['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D']")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--ft_mode', type=str, default='full', help='Fine-tune mode: ["full", "lora"]')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRa rank. (default 8).')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--focal', type=float, default=2.0, help='focal parameter (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of loss for sequence-classifiction. (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of loss for token-classifiction. (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA number (default: 0). ["None",0,1,2,3].')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional training output directory. Defaults to the original finetune path naming rule.')
    parser.add_argument('--save_total_limit', type=int, default=2, help='Maximum number of checkpoints to keep. (default: 2)')
    parser.add_argument('--early_stopping_patience', type=int, default=3, help='Stop training after N evaluation rounds without improvement. Set to 0 to disable.')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.0, help='Minimum metric improvement required by early stopping. (default: 0.0)')
    parser.add_argument('--best_model_info_name', type=str, default='best_model_info.json', help='Filename used to store the best model summary. (default: best_model_info.json)')
    return parser.parse_args()


def build_output_dir(model_name, ft_mode, lora_rank, output_dir=None):
    if output_dir:
        return output_dir
    mod_name = model_name.split('/')[-1]
    folder_name = f"finetune/{mod_name}-full-ft-for-Token-classification"
    if ft_mode == 'lora':
        folder_name = f"finetune/{mod_name}-{ft_mode}-rank-{lora_rank}-ft-for-Sequence-and-Token-classification"
    return folder_name


def train_args_prepare_v2(model_name, learning_rate, batch_size, num_train_epochs, weight_decay, ft_mode, lora_rank, output_dir=None, save_total_limit=2):
    folder_name = build_output_dir(model_name, ft_mode, lora_rank, output_dir=output_dir)
    train_args = TrainingArguments(
        folder_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=save_total_limit,
        push_to_hub=False,)
    return train_args


def trainer_prepare_v2(model, train_args, train_dataset, test_dataset, tokenizer, data_collator, compute_metrics, early_stopping_patience=3, early_stopping_threshold=0.0):
    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ))
    trainer = StatisticsTrainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks,)
    return trainer


def extract_best_epoch_from_log(log_history, best_metric, metric_name='eval_f1'):
    if best_metric is None:
        return None
    for entry in log_history:
        metric_value = entry.get(metric_name)
        if metric_value is None:
            continue
        if abs(metric_value - best_metric) < 1e-12:
            return entry.get('epoch')
    return None


def save_best_model_info(trainer, best_model_info_name='best_model_info.json'):
    output_dir = trainer.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    metric_name = trainer.args.metric_for_best_model
    eval_metric_name = metric_name if str(metric_name).startswith('eval_') else f'eval_{metric_name}'
    best_metric = trainer.state.best_metric
    best_epoch = extract_best_epoch_from_log(trainer.state.log_history, best_metric, metric_name=eval_metric_name)
    best_model_info = {
        'output_dir': output_dir,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric_name': eval_metric_name,
        'best_metric_value': best_metric,
        'best_epoch': best_epoch,
        'global_step': trainer.state.global_step,
        'num_train_epochs': trainer.args.num_train_epochs,
        'save_strategy': str(trainer.args.save_strategy),
        'eval_strategy': str(trainer.args.eval_strategy),
        'early_stopping_patience': next((cb.early_stopping_patience for cb in trainer.callback_handler.callbacks if hasattr(cb, 'early_stopping_patience')), 0),
        'early_stopping_threshold': next((cb.early_stopping_threshold for cb in trainer.callback_handler.callbacks if hasattr(cb, 'early_stopping_threshold')), 0.0),
        'training_args_file': os.path.join(output_dir, 'training_args.json'),
        'training_history_file': os.path.join(output_dir, TRAINING_HISTORY_FILENAME),
        'training_summary_file': os.path.join(output_dir, TRAINING_SUMMARY_FILENAME),
        'log_history': trainer.state.log_history,
    }
    json_path = os.path.join(output_dir, best_model_info_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(best_model_info, f, indent=2, ensure_ascii=False)

    text_path = os.path.join(output_dir, 'best_model_info.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        for key, value in best_model_info.items():
            if key == 'log_history':
                continue
            f.write(f'{key}: {value}\n')
    print(f'Best model information saved to: {json_path}')
    print(f'Best model checkpoint: {trainer.state.best_model_checkpoint}')
    print(f'Best metric ({eval_metric_name}): {best_metric}')
    return best_model_info


def run_training_with_best_model_tracking(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda is None:
        device = 'cpu'
    else:
        if device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    print('Loading model...')
    model, tokenizer, data_collator = model_load(args.model_name, args.num_classes, args.dropout,
                                                 args.ft_mode, args.lora_rank,
                                                 args.focal, args.alpha, args.beta)

    print('Loading data...')
    train_set = MyDataset(os.path.join(args.data_path, 'train.csv'), args.label_path)
    train_dataset = dataset_prepare(train_set, tokenizer)

    val_set = MyDataset(os.path.join(args.data_path, 'val.csv'), args.label_path)
    val_dataset = dataset_prepare(val_set, tokenizer)

    train_args = train_args_prepare_v2(args.model_name,
                                       args.lr,
                                       args.batch_size,
                                       args.epochs,
                                       args.weight_decay,
                                       args.ft_mode,
                                       args.lora_rank,
                                       output_dir=args.output_dir,
                                       save_total_limit=args.save_total_limit)
    trainer = trainer_prepare_v2(model, train_args,
                                 train_dataset=train_dataset, test_dataset=val_dataset,
                                 tokenizer=tokenizer,
                                 data_collator=data_collator,
                                 compute_metrics=compute_metrics,
                                 early_stopping_patience=args.early_stopping_patience,
                                 early_stopping_threshold=args.early_stopping_threshold)
    save_training_args(args, train_args.output_dir)

    print('Begin training...')
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    save_best_model_info(trainer, best_model_info_name=args.best_model_info_name)
    return trainer


def main():
    args = get_parameters_v2()
    run_training_with_best_model_tracking(args)


if __name__ == '__main__':

    main()
