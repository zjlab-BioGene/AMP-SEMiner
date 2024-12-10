import os,sys,re
import argparse
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import EsmForTokenClassification
from transformers import AutoTokenizer, DataCollatorForTokenClassification

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_TRAINING_KEYS = ['input_ids', 'attention_mask']

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

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='AMP token classifer')
    parser.add_argument('--input','-i', type=str, default='../data/AZ107.csv')
    parser.add_argument('--output','-o', type=str, default='./out_prediction.tsv')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training. (default: 4)')
    parser.add_argument('--max_len', type=int, default=300, help='Max sequence length. (default: 300)')
    parser.add_argument('--model_name', type=str, default='../../trainedmodels/ESMforTokenClassification/esm2_650M/epoch20' , help='YOUR_MODEL_PATH.')
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    return args

## prepare model
def get_model(model_name):
    print('Loading model from: %s' % model_name)
    model = EsmForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def prepare_dataset(input_csv, tokenizer, max_len, batch_size):
    print('Loading data from: %s' % input_csv)
    val_set = MyDataset(input_csv)
    sequences, names = val_set.sequences, val_set.names
    inputs = tokenizer(sequences, padding=True, truncation=True, max_length=max_len, return_tensors='pt', add_special_tokens=True)

    eval_dataset = SequenceDataset(inputs, names, sequences)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    return dataloader

def eval_data(dataloader, model):
    print('Predicting...')
    model = model.eval().to(device)
    
    predicts = {}
    proteins = {}
    for _, batch in enumerate(dataloader): 
        torch.cuda.empty_cache()
        names = batch['names']
        sequences = batch['sequences']
        
        ins = {k: v for k, v in batch.items() if k in DATASET_TRAINING_KEYS}
        ins['input_ids'] = ins['input_ids'].to(device)
        ins['attention_mask'] = ins['attention_mask'].to(device)

        outputs = model(**ins)
        logits = outputs.get("logits").detach().cpu()
        torch.cuda.empty_cache()

        for i in range(len(names)):
            seqlen = len(sequences[i])
            pred = logits[i][1:seqlen+1].argmax(dim=1)
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
    model, tokenizer = get_model(args.model_name)
    ## val dataset
    dataloader = prepare_dataset(args.input, tokenizer, args.max_len, args.batch_size)
    predicts, proteins = eval_data(dataloader, model)
    predictions = merge_predictions(predicts, proteins)
    ## output
    with open(args.output,'w') as f:
        f.write('\t'.join(['ProID','AMP','AMPlen','Position','Sequence'])+'\n')
        for k in predictions.keys():
            for a in predictions[k].keys():
                f.write('\t'.join([k, a, str(len(a)),predictions[k][a], proteins[k]])+'\n')

if __name__ == '__main__':

    main()
