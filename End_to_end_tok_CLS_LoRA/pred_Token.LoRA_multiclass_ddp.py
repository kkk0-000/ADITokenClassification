"""
Multi-class token prediction with LoRA + multi-GPU via Accelerate.

Launch examples:
  accelerate launch --num_processes 4 pred_Token.LoRA_multiclass_ddp.py -i input.csv ...
  python pred_Token.LoRA_multiclass_ddp.py -i input.csv ...
"""

import os, sys, re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object
from train_Tokens_LoRA_multiclass_ddp import MyTokensClassification

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

DATASET_TRAINING_KEYS = ['input_ids', 'attention_mask']
CLASS_NAMES = {0: 'background', 1: 'catalytic_triad', 2: 'ADI_insert'}


class MyDataset(Dataset):
    def __init__(self, data_table):
        df = pd.read_csv(data_table, header=None)
        df.columns = ['Class', 'ProId', 'Sequence']
        self.names = df['ProId'].tolist()
        self.sequences = df['Sequence'].tolist()

    def __getitem__(self, index):
        return self.names[index], self.sequences[index]

    def __len__(self):
        return len(self.names)


class TokenizedDataset(Dataset):
    def __init__(self, inputs, names, sequences):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.names = names
        self.sequences = sequences

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'ids': idx}


def get_parameters():
    parser = argparse.ArgumentParser(description='Multi-class token prediction LoRA (DDP)')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='./out_prediction_multiclass.tsv')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--model_name', type=str, default='model_weights/Tok_CLS_LoRA/epoch11')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--class_names', type=str, default='',
                        help='Comma-separated class names')
    args = parser.parse_args()
    return args


def get_model(model_name):
    print('Loading model from: %s' % model_name)
    model = MyTokensClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_dataset(input_csv, tokenizer, max_len, batch_size):
    print('Loading data from: %s' % input_csv)
    raw_set = MyDataset(input_csv)
    sequences, names = raw_set.sequences, raw_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True,
                       max_length=max_len, return_tensors='pt', add_special_tokens=True)
    eval_dataset = TokenizedDataset(inputs, names, sequences)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return dataloader, names, sequences


def eval_data(dataloader, model, names, sequences, accelerator):
    print('Predicting...')
    model.eval()

    local_results = []
    with torch.no_grad():
        for batch in dataloader:
            ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            idxs = batch['ids']

            outputs = model(**ins)
            logits = outputs.get("logits").detach().cpu()

            for j, idx in enumerate(idxs):
                idx = idx.item()
                name = names[idx]
                seq = sequences[idx]
                seqlen = len(seq)
                pred = logits[j][1:seqlen + 1].argmax(dim=1)

                if (pred != 0).any().item():
                    local_results.append({
                        'name': name,
                        'sequence': seq,
                        'pred': pred.tolist(),
                    })

    all_results = gather_object(local_results)
    return all_results


def get_blocks(pred):
    tags, starts, ends = [], [], []
    for i in range(len(pred)):
        if (i == 0) or (pred[i - 1] != pred[i]):
            tags.append(pred[i])
            starts.append(i)
        if (i == len(pred) - 1) or (pred[i + 1] != pred[i]):
            ends.append(i)
    return tags, starts, ends


def merge_predictions(all_results, class_names):
    seen = set()
    predictions = []
    for data in all_results:
        pro_id = data['name']
        if pro_id in seen:
            continue
        seen.add(pro_id)

        seq = data['sequence']
        pred = data['pred']
        tags, starts, ends = get_blocks(pred)

        for i in range(len(tags)):
            cls_id = tags[i]
            if cls_id == 0:
                continue
            fragment = seq[starts[i]:ends[i] + 1]
            cls_name = class_names.get(cls_id, f'class_{cls_id}')
            predictions.append({
                'ProID': pro_id,
                'ClassID': cls_id,
                'ClassName': cls_name,
                'Fragment': fragment,
                'FragLen': len(fragment),
                'Position': f'{starts[i]},{ends[i]}',
                'Sequence': seq,
            })
    return predictions


def main():
    args = get_parameters()
    accelerator = Accelerator()

    class_names = dict(CLASS_NAMES)
    if args.class_names:
        parts = args.class_names.split(',')
        class_names = {i: name.strip() for i, name in enumerate(parts)}

    model, tokenizer = get_model(args.model_name)
    dataloader, names, sequences = prepare_dataset(args.input, tokenizer, args.max_len, args.batch_size)
    model, dataloader = accelerator.prepare(model, dataloader)

    all_results = eval_data(dataloader, model, names, sequences, accelerator)

    if accelerator.is_main_process:
        predictions = merge_predictions(all_results, class_names)
        with open(args.output, 'w') as f:
            header = ['ProID', 'ClassID', 'ClassName', 'Fragment', 'FragLen', 'Position', 'Sequence']
            f.write('\t'.join(header) + '\n')
            for p in predictions:
                f.write('\t'.join([str(p[h]) for h in header]) + '\n')
        n_proteins = len(set(p['ProID'] for p in predictions))
        print(f'Total predictions: {len(predictions)} fragments from {n_proteins} proteins')
        print(f'Output saved to: {args.output}')


if __name__ == '__main__':
    main()
