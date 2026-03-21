# !module load cuda/12.1
# !export LD_LIBRARY_PATH=/YOUR_HOME/miniconda3/envs/gnn/lib:$LD_LIBRARY_PATH

import os,sys,re
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from train_Tokens import MyDataset, compute_binary_metrics

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

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
        return {'labels': self.labels[idx], 'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'ids': idx}

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='AMP token classifer')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--label_path', type=str, default='./labels')
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training. (default: 4)')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length. (default: 300)')
    parser.add_argument('--min_seg_len', type=int, default=5, help='Minimum segament length. (default: 5)')
    parser.add_argument('--model_name', type=str, default='model_weights/Tok_CLS/epoch10' , help='Checkpoint path, training output directory, or best_model_info.json path.')
    args = parser.parse_args()
    return args

def pad_label(lab,max_seq_len=300):
    size = max_seq_len +2 - lab.shape[0]
    new_lab = F.pad(lab, (0, size), mode='constant', value=-100)
    return new_lab.long()

def token2seq_lab(lab, min_seg_len):
    pos_count_best = 0
    pos_count = 0
    n_segments = 0
    for i in range(len(lab)):
        if lab[i] == 1:
            pos_count += 1
            if (i == 0) or (lab[i-1] == 0):
                n_segments += 1
        else:
            if pos_count > pos_count_best:
                pos_count_best = pos_count
                pos_count = 0
            else:
                pos_count = 0
                continue
    if pos_count > pos_count_best:
        pos_count_best = pos_count
        pos_count = 0
    
    tag = 1 if pos_count_best >= min_seg_len else 0
    return tag, n_segments

def process_preds_and_labs(preds,labs,min_seg_len):
    preds_tags, labs_tags = [], []
    for i in range(len(preds)):
        preds_tags.append(token2seq_lab(preds[i][labs[i]!=-100], min_seg_len)[0])
        labs_tags.append(token2seq_lab(labs[i][labs[i]!=-100], min_seg_len)[0])
    return preds_tags, labs_tags

def compute_metrics(preds, labs, probs=None):
    preds = np.asarray(preds)
    labs = np.asarray(labs)
    probs = None if probs is None else np.asarray(probs)
    return compute_binary_metrics(labs, preds, probs=probs)


def load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def find_training_args_file(model_name, resolved_model_name):
    candidates = []
    for candidate in [model_name, resolved_model_name]:
        if not candidate:
            continue
        if os.path.isfile(candidate):
            candidates.append(os.path.join(os.path.dirname(candidate), 'training_args.json'))
        if os.path.isdir(candidate):
            candidates.append(os.path.join(candidate, 'training_args.json'))
            candidates.append(os.path.join(os.path.dirname(candidate.rstrip('/')), 'training_args.json'))
            best_model_info_path = os.path.join(candidate, 'best_model_info.json')
            if os.path.exists(best_model_info_path):
                best_model_info = load_json_if_exists(best_model_info_path)
                if best_model_info is not None:
                    candidates.append(best_model_info.get('training_args_file'))
                    candidates.append(os.path.join(best_model_info.get('output_dir', candidate), 'training_args.json'))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_result_metadata(model_name, resolved_model_name, training_args_file, training_args, protein_metrics, token_metrics):
    return {
        'model_name': model_name,
        'resolved_model_name': resolved_model_name,
        'training_args_file': training_args_file,
        'training_args': training_args,
        'protein_level': protein_metrics,
        'token_level': token_metrics,
    }
    payload.update(token_metrics)
    return payload

def resolve_model_path(model_name):
    if os.path.isfile(model_name) and model_name.endswith('.json'):
        with open(model_name, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        return model_info.get('best_model_checkpoint', model_info.get('output_dir', model_name))

    best_model_info_path = os.path.join(model_name, 'best_model_info.json')
    if os.path.isdir(model_name) and os.path.exists(best_model_info_path):
        with open(best_model_info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        return model_info.get('best_model_checkpoint', model_name)

    return model_name


## prepare model
def get_model(model_name):
    resolved_model_name = resolve_model_path(model_name)
    print('Loading model from: %s' % resolved_model_name)
    model = EsmForTokenClassification.from_pretrained(resolved_model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    return model, tokenizer

def prepare_dataset(data_path, label_path, tokenizer, max_len, batch_size, data_type='val'):
    print('Loading data from: %s' % os.path.join(data_path, data_type+'.csv'))
    val_set = MyDataset(os.path.join(data_path, data_type+'.csv'), label_path)
    labels, sequences, names = [ pad_label(val_set[i][2]) for i in range(len(val_set)) ], val_set.sequences, val_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=max_len+2, return_tensors='pt', add_special_tokens=True)

    eval_dataset = SequenceDataset(inputs, labels, names)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    return dataloader

def eval_data(dataloader, model, min_amp_len):
    print('Predicting...')
    model = model.eval().to(device)
    
    all_preds, all_labs = [], []    ## Protein-level
    all_token_preds, all_token_probs, all_token_labs = [], [], []    ## Token-level
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
            probs = nn.Softmax(dim=1)(logits.reshape((-1,2)).cpu())
            
            if device == 'cuda':
                torch.cuda.empty_cache()
             
            ## Protein-level  
            max_len = preds.shape[1]
            labs = labs[:,:max_len]
            preds_tags, lab_tags = process_preds_and_labs(preds,labs,min_amp_len)
            all_preds += preds_tags
            all_labs += lab_tags
            
            ## Token-level
            labs = labs.reshape(-1)
            preds = preds.reshape(-1)
            preds = preds[labs!=-100]
            probs = probs[labs!=-100][:,1]
            labs = labs[labs!=-100]
            all_token_preds += preds.tolist()
            all_token_labs += labs.tolist()
            all_token_probs += probs.tolist()
    
    ## Protein-level
    metrics_rep = compute_metrics(all_preds, all_labs)
    ## Token-level
    metrics_rep_tok = compute_metrics(all_token_preds, all_token_labs, probs=all_token_probs)
    return metrics_rep, metrics_rep_tok

## main function

def main():
    args = get_parameters()
    model, tokenizer = get_model(args.model_name)

    os.makedirs(args.outdir, exist_ok=True)
    resolved_model_name = resolve_model_path(args.model_name)
    training_args_file = find_training_args_file(args.model_name, resolved_model_name)
    training_args = load_json_if_exists(training_args_file)
    with open(os.path.join(args.outdir, 'eval_metrics.txt'), 'a') as mtx:
        mtx.write('Evaluation on model: %s\n' % os.path.basename(resolved_model_name.rstrip('/')))
        ## val dataset
        # val_dataloader = prepare_dataset(args.data_path, args.label_path,
        #                                 tokenizer, args.max_len, args.batch_size, data_type='test')
        # val_metrics, val_metrics_tok = eval_data(val_dataloader, model, args.min_seg_len)
        # mtx.write('Performance on val_dataset (Protein-level):\n')
        # mtx.write(str(val_metrics)+'\n')
        # mtx.write('Performance on val_dataset (Token-level):\n')
        # mtx.write(str(val_metrics_tok)+'\n')
        
        ## test dataset
        test_dataloader = prepare_dataset(args.data_path, args.label_path,
                                        tokenizer, args.max_len, args.batch_size, data_type='test')
        test_metrics, test_metrics_tok = eval_data(test_dataloader, model, args.min_seg_len)
        mtx.write('Performance on test_dataset (Protein-level):\n')
        mtx.write(str(test_metrics)+'\n')
        mtx.write('Performance on test_dataset (Token-level):\n')
        mtx.write(str(test_metrics_tok)+'\n')
        mtx.write('\n')
        mtx.write('\n')

    test_results_metadata = build_result_metadata(args.model_name, resolved_model_name, training_args_file, training_args, test_metrics, test_metrics_tok)
    with open(os.path.join(args.outdir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics_tok, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, 'protein_level_test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, 'token_level_test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_metrics_tok, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.outdir, 'test_results_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(test_results_metadata, f, indent=2, ensure_ascii=False)
    print('Saved evaluation summary to: %s' % os.path.join(args.outdir, 'test_results.json'))
    
            
            

if __name__ == '__main__':

    main()
