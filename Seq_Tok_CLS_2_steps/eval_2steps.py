# !module load cuda/12.1
# !export LD_LIBRARY_PATH=/home/lwh/miniconda3/envs/gnn/lib:$LD_LIBRARY_PATH

import os,sys,re
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification, EsmForSequenceClassification
from transformers import AutoTokenizer, DataCollatorForTokenClassification

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_TRAINING_KEYS = ['input_ids', 'attention_mask']

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
        # label = torch.from_numpy(np.array([0]*len(sequence)))
        label = torch.from_numpy(np.pad(np.array([0]*len(sequence)),
                                        (1,1), mode='constant', constant_values=-100))
        if prolabel == 1:
            # label = torch.from_numpy(np.load(os.path.join(self.label_path,name+'.npy'))) 
            label = torch.from_numpy(np.pad(np.load(os.path.join(self.label_path,name+'.npy')),
                                (1,1), mode='constant', constant_values=-100))         
        return name, prolabel, label, sequence

    def __len__(self):
        return len(self.names)

class SequenceDataset(Dataset):
    def __init__(self, inputs, labels, names):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.labels = torch.stack(labels)
        self.names = names
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {'labels': self.labels[idx], 'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'names': self.names[idx], 'ids': idx}

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='AMP token classifer')
    parser.add_argument('--data_path', type=str, default='./data_all')
    parser.add_argument('--label_path', type=str, default='/mnt/asustor/wenhui.li/02.AMP/train/esm_token/labels')
    parser.add_argument('--outdir', type=str, default='./')
    parser.add_argument('--datatype', type=str, default='test', help='test or val.')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training. (default: 4)')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length. (default: 300)')
    parser.add_argument('--min_seg_len', type=int, default=5, help='Minimum segament length. (default: 5)')
    parser.add_argument('--model_name1', type=str, default='finetune/esm2_t33_650M_UR50D-ft-for-sequence-classification-full-finetune/epoch20-checkpoint-1859820' , help='YOUR_MODEL_PATH.')
    parser.add_argument('--model_name2', type=str, default='/home/lwh/00.data/AMP/trainedmodels/ftTokenCLS/esm2_t33_650M_UR50D-full-ft-for-Token-classification/epoch15-checkpoint-1394865' , help='YOUR_MODEL_PATH.')
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

def compute_metrics(preds, labs):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    return metric.compute(predictions=preds, references=labs)

def report_pred(preds, labs):
    positives, negatives = 0, 0 
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(preds)):
        if labs[i] == 1:
            positives += 1
            if preds[i] == labs[i]:
                tp += 1
            else:
                fn += 1
        else:
            negatives += 1
            if preds[i] == labs[i]:
                tn += 1
            else:
                fp += 1
    return {'Total': positives+negatives , 'Positive': positives, 'Negative': negatives, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

## prepare model
def get_model(model_name1, model_name2):
    print('Loading SequenceCLS model from: %s' % model_name1)
    model1 = EsmForSequenceClassification.from_pretrained(model_name1)
    tokenizer = AutoTokenizer.from_pretrained(model_name1)
    
    print('Loading TokenCLS model from: %s' % model_name2)
    model2 = EsmForTokenClassification.from_pretrained(model_name2)
    return model1, model2, tokenizer

def prepare_dataset(data_path, label_path, tokenizer, max_len, batch_size, data_type='test'):
    print('Loading data from: %s' % os.path.join(data_path, data_type+'.csv'))
    val_set = MyDataset(os.path.join(data_path, data_type+'.csv'), label_path)
    labels, sequences, names = [ pad_label(val_set[i][2]) for i in range(len(val_set)) ], val_set.sequences, val_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=max_len+2, return_tensors='pt', add_special_tokens=True)

    eval_dataset = SequenceDataset(inputs, labels, names)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    return dataloader

def eval_data(dataloader, model1, model2, min_amp_len):
    print('Predicting...')
    model1 = model1.eval().to(device)
    model2 = model2.eval().to(device)
    
    all_preds, all_labs = [], []    ## Protein-level
    all_token_preds, all_token_labs = [], []    ## Token-level
    with torch.no_grad():
        for _, batch in enumerate(dataloader): 
            torch.cuda.empty_cache()
            ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
            labs = batch['labels']
            ins['input_ids'] = ins['input_ids'].to(device)
            ins['attention_mask'] = ins['attention_mask'].to(device)

            out1 = model1(**ins)
            out2 = model2(**ins)
            a = torch.argmax(out1[0], axis=1).cpu()
            preds = torch.argmax(out2[0], axis=2).cpu()
            for i in range(a.shape[0]):
                if a[i] == 0:
                    preds[i] *= 0
            
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
            labs = labs[labs!=-100]
            all_token_preds += preds.tolist()
            all_token_labs += labs.tolist()
    
    ## Protein-level    
    metrics_rep = compute_metrics(all_preds,all_labs)
    reports = report_pred(all_preds,all_labs)
    ## Token-level
    metrics_rep_tok = compute_metrics(all_token_preds,all_token_labs)
    reports_tok = report_pred(all_token_preds,all_token_labs)   
    
    return metrics_rep, reports, metrics_rep_tok, reports_tok

## main function

def main():
    args = get_parameters()
    model1, model2, tokenizer = get_model(args.model_name1, args.model_name2)
    
    with open(os.path.join(args.outdir, 'eval_metrics.txt'), 'a') as mtx:
        mtx.write('Evaluation on model: %s and %s\n' % (args.model_name1, args.model_name2))
        
        ## test dataset
        test_dataloader = prepare_dataset(args.data_path, args.label_path,
                                        tokenizer, args.max_len, args.batch_size, data_type=args.datatype)
        test_metrics, test_reports, test_metrics_tok, test_reports_tok = eval_data(test_dataloader, model1, model2, args.min_seg_len)
        mtx.write('Performance on test_dataset (Protein-level):\n')
        mtx.write(str(test_metrics)+'\n')
        mtx.write(str(test_reports)+'\n')
        mtx.write('Performance on test_dataset (Token-level):\n')
        mtx.write(str(test_metrics_tok)+'\n')
        mtx.write(str(test_reports_tok)+'\n')
        mtx.write('\n')
        mtx.write('\n')

if __name__ == '__main__':

    main()
