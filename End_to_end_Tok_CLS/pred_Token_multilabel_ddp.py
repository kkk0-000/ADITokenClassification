"""
Multi-LABEL token prediction with multi-GPU via Accelerate.

Launch examples:
  accelerate launch --num_processes 4 pred_Token_multilabel_ddp.py -i input.csv --num_labels 3 ...
  python pred_Token_multilabel_ddp.py -i input.csv --num_labels 3 ...
"""

import os, sys, re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'

DATASET_TRAINING_KEYS = ['input_ids', 'attention_mask']
LABEL_NAMES = {0: 'background', 1: 'catalytic_triad', 2: 'ADI_insert'}


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
    parser = argparse.ArgumentParser(description='Multi-label token prediction (DDP)')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default='./out_prediction_multilabel.tsv')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--model_name', type=str, default='model_weights/Tok_CLS/epoch15')
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--label_names', type=str, default='',
                        help='Comma-separated label names')
    args = parser.parse_args()
    return args


def get_model(model_name):
    model = EsmForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_dataset(input_csv, tokenizer, max_len, batch_size):
    raw_set = MyDataset(input_csv)
    sequences, names = raw_set.sequences, raw_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True,
                       max_length=max_len, return_tensors='pt', add_special_tokens=True)
    ds = TokenizedDataset(inputs, names, sequences)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return dl, names, sequences


def eval_data(dataloader, model, names, sequences, threshold, num_labels, accelerator):
    model.eval()
    local_results = []
    with torch.no_grad():
        for batch in dataloader:
            ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            idxs = batch['ids']

            outputs = model(**ins)
            logits = outputs.logits.detach().cpu()

            for j, idx in enumerate(idxs):
                idx = idx.item()
                name = names[idx]
                seq = sequences[idx]
                seqlen = len(seq)
                token_logits = logits[j][1:seqlen + 1]  # (seqlen, num_labels)
                probs = torch.sigmoid(token_logits)
                preds = (probs > threshold).int()  # (seqlen, num_labels)

                if preds.any().item():
                    local_results.append({
                        'name': name,
                        'sequence': seq,
                        'preds': preds.numpy().tolist(),
                        'probs': probs.numpy().tolist(),
                    })

    all_results = gather_object(local_results)
    return all_results


def get_label_segments(preds_per_label, label_id):
    """Extract contiguous segments of 1s for a specific label."""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(preds_per_label):
        if v == 1 and not in_seg:
            start = i
            in_seg = True
        elif v == 0 and in_seg:
            segments.append((label_id, start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((label_id, start, len(preds_per_label) - 1))
    return segments


def merge_predictions(all_results, label_names, num_labels):
    seen = set()
    predictions = []
    for data in all_results:
        pro_id = data['name']
        if pro_id in seen:
            continue
        seen.add(pro_id)

        seq = data['sequence']
        preds = np.array(data['preds'])  # (seqlen, num_labels)

        for label_id in range(num_labels):
            if label_id == 0:
                continue
            segments = get_label_segments(preds[:, label_id], label_id)
            for _, start, end in segments:
                fragment = seq[start:end + 1]
                lbl_name = label_names.get(label_id, f'label_{label_id}')

                overlap_labels = set()
                for pos in range(start, end + 1):
                    for other_id in range(num_labels):
                        if other_id != label_id and preds[pos, other_id] == 1:
                            overlap_labels.add(label_names.get(other_id, f'label_{other_id}'))

                predictions.append({
                    'ProID': pro_id,
                    'LabelID': label_id,
                    'LabelName': lbl_name,
                    'Fragment': fragment,
                    'FragLen': len(fragment),
                    'Position': f'{start},{end}',
                    'OverlapLabels': ';'.join(sorted(overlap_labels)) if overlap_labels else 'none',
                    'Sequence': seq,
                })

    return predictions


def main():
    args = get_parameters()
    accelerator = Accelerator()

    label_names = dict(LABEL_NAMES)
    if args.label_names:
        parts = args.label_names.split(',')
        label_names = {i: name.strip() for i, name in enumerate(parts)}

    model, tokenizer = get_model(args.model_name)
    dataloader, names, sequences = prepare_dataset(args.input, tokenizer, args.max_len, args.batch_size)
    model, dataloader = accelerator.prepare(model, dataloader)

    all_results = eval_data(dataloader, model, names, sequences,
                            args.threshold, args.num_labels, accelerator)

    if accelerator.is_main_process:
        predictions = merge_predictions(all_results, label_names, args.num_labels)
        header = ['ProID', 'LabelID', 'LabelName', 'Fragment', 'FragLen',
                  'Position', 'OverlapLabels', 'Sequence']
        with open(args.output, 'w') as f:
            f.write('\t'.join(header) + '\n')
            for p in predictions:
                f.write('\t'.join([str(p[h]) for h in header]) + '\n')
        n_proteins = len(set(p['ProID'] for p in predictions))
        print(f'Total: {len(predictions)} fragments from {n_proteins} proteins')
        print(f'Output: {args.output}')


if __name__ == '__main__':
    main()
