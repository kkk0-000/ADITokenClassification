import os,sys,re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from train_Tokens_multiclass import MyDataset

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    parser = argparse.ArgumentParser(description='Multi-class token classifier evaluation')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='model_weights/Tok_CLS/epoch10',
                        help='YOUR_MODEL_PATH.')
    args = parser.parse_args()
    return args


def pad_label(lab, max_seq_len=300):
    size = max_seq_len + 2 - lab.shape[0]
    new_lab = F.pad(lab, (0, size), mode='constant', value=-100)
    return new_lab.long()


def compute_metrics(preds, labs, num_classes):
    results = {
        "accuracy": accuracy_score(labs, preds),
        "f1_macro": f1_score(labs, preds, average='macro', zero_division=0),
        "precision_macro": precision_score(labs, preds, average='macro', zero_division=0),
        "recall_macro": recall_score(labs, preds, average='macro', zero_division=0),
        "f1_weighted": f1_score(labs, preds, average='weighted', zero_division=0),
    }
    for c in range(num_classes):
        binary_preds = [1 if p == c else 0 for p in preds]
        binary_labs = [1 if l == c else 0 for l in labs]
        results[f"f1_class{c}"] = f1_score(binary_labs, binary_preds, zero_division=0)
        results[f"precision_class{c}"] = precision_score(binary_labs, binary_preds, zero_division=0)
        results[f"recall_class{c}"] = recall_score(binary_labs, binary_preds, zero_division=0)
    return results
def compute_metrics(preds, labs):
    return {
        "accuracy": accuracy_score(labs, preds),
        "f1": f1_score(labs, preds, zero_division=0),
        "precision": precision_score(labs, preds, zero_division=0),
        "recall": recall_score(labs, preds, zero_division=0),
    }
def compute_roc_auc(probs, labs):
    try:
        return {"roc_auc": roc_auc_score(labs, probs)}
    except ValueError:
        return {"roc_auc": 0.0}

def report_pred_multiclass(preds, labs, num_classes):
    report = {}
    report['Total'] = len(preds)
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
    model = EsmForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_dataset(data_path, label_path, tokenizer, max_len, batch_size, data_type='test'):
    print('Loading data from: %s' % os.path.join(data_path, data_type+'.csv'))
    val_set = MyDataset(os.path.join(data_path, data_type+'.csv'), label_path)
    labels = [pad_label(val_set[i][2]) for i in range(len(val_set))]
    sequences, names = val_set.sequences, val_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True,
                       max_length=max_len+2, return_tensors='pt', add_special_tokens=True)
    eval_dataset = SequenceDataset(inputs, labels, names)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    return dataloader


def eval_data(dataloader, model, num_classes):
    print('Predicting...')
    model = model.eval().to(device)

    all_token_preds, all_token_labs = [], []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            labs = batch['labels']
            ins['input_ids'] = ins['input_ids'].to(device)
            ins['attention_mask'] = ins['attention_mask'].to(device)

            outputs = model(**ins)
            logits = outputs.get("logits")
            preds = torch.argmax(logits, dim=2).cpu()

            if device == 'cuda':
                torch.cuda.empty_cache()

            max_len = preds.shape[1]
            labs = labs[:, :max_len]

            labs_flat = labs.reshape(-1)
            preds_flat = preds.reshape(-1)
            mask = labs_flat != -100
            all_token_preds += preds_flat[mask].tolist()
            all_token_labs += labs_flat[mask].tolist()

    metrics_tok = compute_metrics(all_token_preds, all_token_labs, num_classes)
    reports_tok = report_pred_multiclass(all_token_preds, all_token_labs, num_classes)

    return metrics_tok, reports_tok


def main():
    args = get_parameters()
    model, tokenizer = get_model(args.model_name)

    with open(os.path.join(args.outdir, 'eval_metrics_multiclass.txt'), 'a') as mtx:
        mtx.write('Evaluation on model: %s\n' % args.model_name.split('/')[-1])

        test_dataloader = prepare_dataset(args.data_path, args.label_path,
                                          tokenizer, args.max_len, args.batch_size, data_type='test')
        test_metrics_tok, test_reports_tok = eval_data(test_dataloader, model, args.num_classes)
        mtx.write('Performance on test_dataset (Token-level, multi-class):\n')
        mtx.write(str(test_metrics_tok)+'\n')
        mtx.write(str(test_reports_tok)+'\n')
        mtx.write('\n\n')

    print('Token-level metrics:', test_metrics_tok)
    print('Token-level report:', test_reports_tok)

if __name__ == '__main__':
    main()
