"""
Multi-class token classification evaluation with LoRA + multi-GPU via Accelerate.

Launch examples:
  accelerate launch --num_processes 4 eval_Tokens.CE_LoRA_multiclass_ddp.py --num_classes 3 ...
  python eval_Tokens.CE_LoRA_multiclass_ddp.py --num_classes 3 ...
"""

import os, sys, re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification
from transformers import AutoTokenizer
from accelerate import Accelerator

from train_Tokens_LoRA_multiclass_ddp import MyDataset, MyTokensClassification

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

DATASET_TRAINING_KEYS = ['input_ids', 'attention_mask']


class SequenceDataset(Dataset):
    def __init__(self, inputs, labels, names):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels = torch.stack(labels)
        self.names = names

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'labels': self.labels[idx],
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'ids': idx}


def get_parameters():
    parser = argparse.ArgumentParser(description='Multi-class token eval LoRA (DDP)')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='model_weights/Tok_CLS_LoRA/epoch11')
    args = parser.parse_args()
    return args


def pad_label(lab, max_seq_len=300):
    size = max_seq_len + 2 - lab.shape[0]
    new_lab = F.pad(lab, (0, size), mode='constant', value=-100)
    return new_lab.long()


def compute_metrics_fn(preds, labs, probs, num_classes):
    results = {
        "accuracy": accuracy_score(labs, preds),
        "f1_macro": f1_score(labs, preds, average='macro', zero_division=0),
        "precision_macro": precision_score(labs, preds, average='macro', zero_division=0),
        "recall_macro": recall_score(labs, preds, average='macro', zero_division=0),
        "f1_weighted": f1_score(labs, preds, average='weighted', zero_division=0),
    }
    try:
        if num_classes == 2:
            results["roc_auc"] = roc_auc_score(labs, [p[1] for p in probs])
        else:
            results["roc_auc"] = roc_auc_score(labs, probs, multi_class='ovr', average='macro')
    except ValueError:
        results["roc_auc"] = 0.0

    for c in range(num_classes):
        binary_preds = [1 if p == c else 0 for p in preds]
        binary_labs = [1 if l == c else 0 for l in labs]
        results[f"f1_class{c}"] = f1_score(binary_labs, binary_preds, zero_division=0)
        results[f"precision_class{c}"] = precision_score(binary_labs, binary_preds, zero_division=0)
        results[f"recall_class{c}"] = recall_score(binary_labs, binary_preds, zero_division=0)
    return results


def report_pred_multiclass(preds, labs, num_classes):
    report = {'Total': len(preds)}
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labs) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labs) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labs) if p != c and l == c)
        tn = sum(1 for p, l in zip(preds, labs) if p != c and l != c)
        report[f'Class{c}_TP'] = tp
        report[f'Class{c}_FP'] = fp
        report[f'Class{c}_FN'] = fn
        report[f'Class{c}_TN'] = tn
    return report


def get_model(model_name):
    print('Loading model from: %s' % model_name)
    model = MyTokensClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_dataset(data_path, label_path, tokenizer, max_len, batch_size, data_type='test'):
    print('Loading data from: %s' % os.path.join(data_path, data_type + '.csv'))
    val_set = MyDataset(os.path.join(data_path, data_type + '.csv'), label_path)
    labels = [pad_label(val_set[i][2], max_len) for i in range(len(val_set))]
    sequences, names = val_set.sequences, val_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True,
                       max_length=max_len + 2, return_tensors='pt', add_special_tokens=True)
    eval_dataset = SequenceDataset(inputs, labels, names)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def eval_data(dataloader, model, num_classes, accelerator):
    print('Predicting...')
    model.eval()

    all_token_preds, all_token_labs, all_token_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            labs = batch['labels']

            outputs = model(**ins)
            logits = outputs.get("logits")
            preds = torch.argmax(logits, dim=2)
            probs = torch.softmax(logits, dim=2)

            preds_gathered = accelerator.gather_for_metrics(preds)
            labs_gathered = accelerator.gather_for_metrics(labs)
            probs_gathered = accelerator.gather_for_metrics(probs)

            preds_g = preds_gathered.cpu()
            labs_g = labs_gathered.cpu()
            probs_g = probs_gathered.cpu()

            max_len = preds_g.shape[1]
            labs_g = labs_g[:, :max_len]

            labs_flat = labs_g.reshape(-1)
            preds_flat = preds_g.reshape(-1)
            probs_flat = probs_g.reshape(-1, num_classes)
            mask = labs_flat != -100
            all_token_preds += preds_flat[mask].tolist()
            all_token_labs += labs_flat[mask].tolist()
            all_token_probs += probs_flat[mask].tolist()

    metrics_tok = compute_metrics_fn(all_token_preds, all_token_labs, all_token_probs, num_classes)
    reports_tok = report_pred_multiclass(all_token_preds, all_token_labs, num_classes)

    return metrics_tok, reports_tok


def main():
    args = get_parameters()
    accelerator = Accelerator()

    model, tokenizer = get_model(args.model_name)
    test_dataloader = prepare_dataset(args.data_path, args.label_path,
                                      tokenizer, args.max_len, args.batch_size, data_type='test')

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    test_metrics_tok, test_reports_tok = eval_data(test_dataloader, model, args.num_classes, accelerator)

    if accelerator.is_main_process:
        with open(os.path.join(args.outdir, 'eval_metrics_multiclass.txt'), 'a') as mtx:
            mtx.write('Evaluation on model: %s\n' % args.model_name)
            mtx.write('Performance on test_dataset (Token-level, multi-class):\n')
            mtx.write(str(test_metrics_tok) + '\n')
            mtx.write(str(test_reports_tok) + '\n')
            mtx.write('\n\n')
        print('Token-level metrics:', test_metrics_tok)
        print('Token-level report:', test_reports_tok)


if __name__ == '__main__':
    main()
