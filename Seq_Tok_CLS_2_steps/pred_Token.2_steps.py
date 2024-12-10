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

root_path = '/mnt/asustor/wenhui.li/02.AMP/train/2_step_train'
if root_path not in sys.path:
    sys.path.append(root_path)


class SequenceDataset(Dataset):
    def __init__(self, inputs, names, sequences):
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']
        self.names = names
        self.sequences = sequences
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'ids': idx, 'names': self.names[idx], 'sequences': self.sequences[idx]}

class MyDataset(Dataset):

    def __init__(self, data_table):
        df = pd.read_csv(data_table,header=None)
        df.columns = ['Class','ProId','Sequence']
        self.names = df['ProId'].tolist()
        self.sequences = df['Sequence'].tolist()

    def __getitem__(self, index):
        name = self.names[index]
        sequence = self.sequences[index]
        label = torch.from_numpy(np.pad(np.array([0]*len(sequence)),
                                 (1,1), mode='constant', constant_values=-100)) 
        return name, label, sequence

    def __len__(self):
        return len(self.names)

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='AMP token classifer')
    parser.add_argument('--input','-i', type=str, default='../data/AZ107.csv')
    parser.add_argument('--output','-o', type=str, default='./out_prediction.tsv')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training. (default: 4)')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length. (default: 300)')
    parser.add_argument('--model_name1', type=str, default='/mnt/asustor/wenhui.li/02.AMP/train/2_step_train/finetune/esm2_t33_650M_UR50D-ft-for-sequence-classification-full-finetune/epoch20-checkpoint-1859820' , help='YOUR_MODEL_PATH.')
    parser.add_argument('--model_name2', type=str, default='/mnt/asustor/wenhui.li/02.AMP/train/2_step_train/finetune/esm2_t33_650M_UR50D-ft-for-Token-classification-full-finetune/epoch50-checkpoint-41550' , help='YOUR_MODEL_PATH.')

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

# def compute_roc_auc(probs, labs):
#     metric = evaluate.load('roc_auc')
#     return metric.compute(prediction_scores=probs, references=labs)

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


def prepare_dataset(input_csv, tokenizer, max_len, batch_size):
    print('Loading data from: %s' % input_csv)
    val_set = MyDataset(input_csv)
    sequences, names = val_set.sequences, val_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=max_len, return_tensors='pt', add_special_tokens=True)

    eval_dataset = SequenceDataset(inputs, names, sequences)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    return dataloader

def eval_data(dataloader, model1, model2):
    print('Predicting...')
    model1 = model1.eval().to(device)
    model2 = model2.eval().to(device)
    
    predicts = {}
    proteins = {}
    for _, batch in enumerate(dataloader): 
        torch.cuda.empty_cache()
        names = batch['names']
        sequences = batch['sequences']
        
        ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
        ins['input_ids'] = ins['input_ids'].to(device)
        ins['attention_mask'] = ins['attention_mask'].to(device)

        out1 = model1(**ins)
        out2 = model2(**ins)
        a = torch.argmax(out1[0], axis=1).cpu()
        preds = torch.argmax(out2[0], axis=2).cpu()
        for i in range(a.shape[0]):
            if a[i] == 0:
                    preds[i] *= 0

        for i in range(len(names)):
            seqlen = len(sequences[i])
            pred = preds[i]
            if pred.nonzero().size(0) >= 5:
                proteins[names[i]] = sequences[i]
                predicts[names[i]] = pred  

    return predicts, proteins

def get_blocks(pred):
    tags, starts, ends = [], [], []
    for i in range(pred.shape[0]):
        if (i==0) or (pred[i-1] != pred[i]):
            tags.append(int(pred[i]))
            starts.append(i)
        if (i==pred.shape[0]-1) or (pred[i+1] != pred[i]):
            ends.append(i)
    return torch.tensor(tags), starts, ends

def merge_predictions(predicts, proteins):
    predictions = {}
    for k in predicts.keys():
        predictions[k] = {}
        tags, starts, ends = get_blocks(predicts[k])
        for i in range(len(tags)):
            if tags[i] == 1:
                predictions[k][proteins[k][starts[i]:ends[i]+1]] = str(starts[i])+','+str(ends[i])
    return predictions

## main function

def main():
    args = get_parameters()
    model1, model2, tokenizer = get_model(args.model_name1, args.model_name2)
    
    ## val dataset
    dataloader = prepare_dataset(args.input, tokenizer, args.max_len, args.batch_size)
    predicts, proteins = eval_data(dataloader, model1, model2)
    predictions = merge_predictions(predicts, proteins)
    ## output
    with open(args.output,'w') as f:
        f.write('\t'.join(['ProID','AMP','AMPlen','Position','Sequence'])+'\n')
        for k in predictions.keys():
            for a in predictions[k].keys():
                f.write('\t'.join([k, a, str(len(a)),predictions[k][a], proteins[k]])+'\n')

if __name__ == '__main__':

    main()