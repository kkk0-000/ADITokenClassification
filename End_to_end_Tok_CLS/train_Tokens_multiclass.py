import os,sys,re
import copy
import math
from typing import List, Optional, Tuple, Union
import argparse
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import torch
import datasets
import torch.nn as nn
from torch.nn import Dropout, Linear, CrossEntropyLoss

import torch.utils.checkpoint
from torch.utils.data import Dataset
from transformers import EsmModel, EsmPreTrainedModel, EsmForTokenClassification
from transformers.models.esm.modeling_esm import EsmClassificationHead
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
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
        label = torch.from_numpy(np.pad(np.array([0]*len(sequence)),
                                        (1,1), mode='constant', constant_values=-100))
        if prolabel >= 1:
            label = torch.from_numpy(np.pad(np.load(os.path.join(self.label_path, name+'.npy')).astype(np.int64),
                                            (1,1), mode='constant', constant_values=-100))
        return name, prolabel, label, sequence

    def __len__(self):
        return len(self.names)

    def get_num_samples_per_class(self):
        return torch.bincount(torch.tensor(self.prolabels)).long().tolist()

def get_class_weight(num_samples_per_class):
    normalized_weights = [1/(count/sum(num_samples_per_class)) for count in num_samples_per_class]
    print(normalized_weights)
    return torch.FloatTensor(normalized_weights)

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='Multi-class token classifier')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t30_150M_UR50D',
                        help="ESM2 model name or path")
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_classes', type=int, default=3, help='number of token classes (default: 3)')
    parser.add_argument('--ft_mode', type=str, default='full', help='Fine-tune mode: ["full", "lora"]')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRa rank. (default 8).')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--focal', type=float, default=2.0, help='focal parameter (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of loss for sequence-classifiction. (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of loss for token-classifiction. (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA number (default: 0). ["None",0,1,2,3].')
    args = parser.parse_args()
    return args

## Model preparing
def model_load(model_name, num_labels, dropout, ft_mode, lora_rank, focal, alpha, beta):
    model = EsmForTokenClassification.from_pretrained(model_name, num_labels=num_labels,
                                                      hidden_dropout_prob=dropout,
                                                      attention_probs_dropout_prob=dropout)
    if ft_mode == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            bias="none",
            use_rslora=True,
            r=lora_rank,
            lora_alpha=16*(lora_rank**.5),
            lora_dropout=0.05,
            target_modules=[
                "query", "key", "value",
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

def train_args_prepare(model_name, learning_rate, batch_size, num_train_epochs,
                       weight_decay, ft_mode, lora_rank, num_classes):
    mod_name = model_name.split('/')[-1]
    folder_name = f"finetune/{mod_name}-full-ft-for-Token-multiclass-{num_classes}"
    if ft_mode == 'lora':
        folder_name = f"finetune/{mod_name}-{ft_mode}-rank-{lora_rank}-ft-for-Token-multiclass-{num_classes}"
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
        metric_for_best_model='f1_macro',
        push_to_hub=False,)
    return train_args

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    num_classes = predictions.shape[-1]
    labels = labels.reshape((-1))
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro', zero_division=0),
        "precision_macro": precision_score(labels, predictions, average='macro', zero_division=0),
        "recall_macro": recall_score(labels, predictions, average='macro', zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average='weighted', zero_division=0),
        "precision_weighted": precision_score(labels, predictions, average='weighted', zero_division=0),
        "recall_weighted": recall_score(labels, predictions, average='weighted', zero_division=0),
    }

## Data preparing
def dataset_prepare(MyDataset_obj, tokenizer):
    token_labels = [MyDataset_obj[i][2].numpy() for i in range(len(MyDataset_obj))]
    sequences = MyDataset_obj.sequences
    tokenized = tokenizer(sequences)
    dataset = datasets.Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", token_labels)
    return dataset

## Trainer preparing
def trainer_prepare(model, train_args, train_dataset, test_dataset,
                    tokenizer, data_collator, compute_metrics):
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,)
    return trainer

## main
def main():
    args = get_parameters()

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

    train_args = train_args_prepare(args.model_name,
                                    args.lr,
                                    args.batch_size,
                                    args.epochs,
                                    args.weight_decay,
                                    args.ft_mode,
                                    args.lora_rank,
                                    args.num_classes)
    trainer = trainer_prepare(model, train_args,
                              train_dataset=train_dataset, test_dataset=val_dataset,
                              tokenizer=tokenizer,
                              data_collator=data_collator,
                              compute_metrics=compute_metrics,)

    print('Begin training...')
    trainer.train()

if __name__ == '__main__':
    main()
