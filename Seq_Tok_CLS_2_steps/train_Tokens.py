# !module load cuda/12.1
# !export LD_LIBRARY_PATH=/home/lwh/miniconda3/envs/gnn/lib:$LD_LIBRARY_PATH

import os,sys,re
import copy
import math
from typing import List, Optional, Tuple, Union
import argparse
import random
import time
import evaluate
import numpy as np
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

@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class AMPTokensClassification(EsmForTokenClassification):
    def __init__(self, config, freeze=True):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.esm = EsmModel(config, add_pooling_layer=False)
        if freeze:
            for param in self.esm.base_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



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
        label = torch.from_numpy(np.array([0]*len(sequence)))
        if prolabel == 1:
            label = torch.from_numpy(np.load(os.path.join(self.label_path,name+'.npy')))        
        return name, prolabel, label, sequence

    def __len__(self):
        return len(self.names)
    
    def get_num_samples_per_class(self):
        return torch.bincount(torch.tensor(self.prolabels)).long().tolist()

## Get parameters
def get_parameters():
    parser = argparse.ArgumentParser(description='AMP token classifer')
    parser.add_argument('--label_path', type=str, default='/home/lwh/00.data/AMP/train/esm_token_freeze/labels')
    parser.add_argument('--data_path', type=str, default='/home/lwh/00.data/AMP/train/esm_token_freeze/testdata')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--model_name', type=str, default='/home/lwh/00.data/AMP/basemodels/esm2_t30_150M_UR50D' , help="['facebook/esm2_t6_8M_UR50D', 'facebook/esm2_t12_35M_UR50D', 'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D']")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--ft_mode', type=str, default='freeze', help='Fine-tune mode: ["full", "lora", "freeze"]')
    parser.add_argument('--lora_rank', type=int, default=8,help='LoRa rank. (default 8).')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--focal', type=float, default=2.0, help='focal parameter (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of loss for sequence-classifiction. (default: 1.0)')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight of loss for token-classifiction. (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--cuda', type=int, default=1, help='CUDA number (default: 0). ["None",0,1,2,3].')
    args = parser.parse_args()
    return args

## Model preparing
def model_load(model_name,num_labels,ft_mode,lora_rank,focal,alpha,beta):
    # model = EsmForSequenceTokenClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=torch.bfloat16,
    #                                                           focal=focal, alpha=alpha, beta=beta)
    print("model_name", model_name)
    if ft_mode == "freeze":
        model = AMPTokensClassification.from_pretrained(model_name, num_labels=num_labels, freeze=True)
    elif ft_mode == 'lora':
        model = AMPTokensClassification.from_pretrained(model_name, num_labels=num_labels, freeze=False)
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
    else:
        model = AMPTokensClassification.from_pretrained(model_name, num_labels=num_labels, freeze=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return model, tokenizer, data_collator

def train_args_prepare(model_name, learning_rate, batch_size, num_train_epochs, weight_decay, ft_mode, lora_rank):
    mol_name = model_name.split('/')[-1]
    folder_name = f"finetune/{mol_name}-ft-for-Token-classification-full-finetune"
    if ft_mode == 'lora':
        folder_name = f"finetune/{mol_name}-{ft_mode}-rank-{lora_rank}-ft-for-Sequence-and-Token-classification"
    train_args = TrainingArguments(
        folder_name,
        evaluation_strategy = "epoch",
        # eval_steps = 50,
        save_total_limit=5,
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

def compute_metrics(eval_pred):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    predictions, labels = eval_pred
    labels = labels.reshape((-1))
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    predictions = predictions[labels!=-100]
    labels = labels[labels!=-100]
    return metric.compute(predictions=predictions, references=labels)

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

## main
def main():
    args = get_parameters()
    
    ## Seed setting
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    ## Device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda == None:
        device = 'cpu'
    else:   
        if device == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda) 
    
    ## Load model
    print('Loading model...')
    model, tokenizer, data_collator = model_load(args.model_name, args.num_classes,
                                                 args.ft_mode, args.lora_rank, 
                                                 args.focal, args.alpha, args.beta)

    ## Data processing
    print('Loading data...')
    train_set = MyDataset(os.path.join(args.data_path,'train.csv'), args.label_path)
    train_dataset = dataset_prepare(train_set, tokenizer)
    
    val_set = MyDataset(os.path.join(args.data_path,'val.csv'), args.label_path)
    val_dataset = dataset_prepare(val_set, tokenizer)
    
    ## Trainer
    train_args = train_args_prepare(args.model_name, 
                                    args.lr, 
                                    args.batch_size, 
                                    args.epochs, 
                                    args.weight_decay, 
                                    args.ft_mode,
                                    args.lora_rank)
    trainer = trainer_prepare(model, train_args, 
                              train_dataset=train_dataset, test_dataset=val_dataset, 
                              tokenizer=tokenizer, 
                              data_collator=data_collator, 
                              compute_metrics=compute_metrics,)

    ## Training
    print('Begin training...')
    trainer.train()
        
if __name__ == '__main__':

    main()
