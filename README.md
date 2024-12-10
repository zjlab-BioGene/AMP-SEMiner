# AMP-SEMiner: <u>A</u>nti<u>M</u>icrobial <u>P</u>eptide <u>S</u>tructural <u>E</u>volution <u>M</u>iner

## Introduction

This repository contains the custom code for **AMP-SEMiner**, a robust and comprehensive AI framework designed for residue-level AMP discovery. The framework accurately identifies specific residues within a protein as either AMP or non-AMP.

![Schematic diagram of the AMP-SEMiner framework](Figure_1.png)

## Model weights & example data


## Environment
AMP-SEMiner is run on Python 3.10 and PyTorch 2.1.2. You can build a conda environment for AMP-SEMiner using this [script](https://github.com/zjlab-BioGene/AMP-SEMiner/blob/main/scripts/env_install.sh).

## Run AMP prediction

The model weights of AMP-SEMiner and minimum datasets to run AMP-SEMiner are available in [Zenodo](https://zenodo.org/records/14348290) (DOI: 10.5281/zenodo.14348290). A small example dataset are also provided:

[Tok_CLS.tar.gz](https://zenodo.org/records/14348290/files/Tok_CLS.tar.gz)

[Tok_CLS_LoRA.tar.gz](https://zenodo.org/records/14348290/files/Tok_CLS_LoRA.tar.gz)

[2_steps.tar.gz](https://zenodo.org/records/14348290/files/2_steps.tar.gz)

[example_dataset.tar.gz](https://zenodo.org/records/14348290/files/example_dataset.tar.gz)

<pre>
mkdir model_weights
cd model_weights

## download model weights
wget https://zenodo.org/records/14348290/files/Tok_CLS.tar.gz
tar -zxvf Tok_CLS.tar.gz

wget https://zenodo.org/records/14348290/files/Tok_CLS_LoRA.tar.gz
tar -zxvf Tok_CLS_LoRA.tar.gz

cd -

## download example datasets
wget https://zenodo.org/records/14348290/files/example_dataset.tar.gz
tar -zxvf example_dataset.tar.gz
</pre>

<pre>
mkdir example_out

#conda activate ampseminer
python End_to_end_Tok_CLS/pred_Token.py \
    --input example_dataset/test_dataset.csv \
    --output example_out/test.out_pred.tsv \
    --model_name model_weights/Tok_CLS/epoch15 \
    --batch_size 4 \
    --max_len 300
</pre>

## Run training

Example datasets for AMP model trainining:

[example_data.tar.gz](https://zenodo.org/records/14348290/files/example_data.tar.gz)

[example_labels.tar.gz](https://zenodo.org/records/14348290/files/example_labels.tar.gz)

<pre>
## download example datasets for AMP model training

wget https://zenodo.org/records/14348290/files/example_data.tar.gz
tar -zxvf example_data.tar.gz

wget https://zenodo.org/records/14348290/files/example_labels.tar.gz
tar -zxvf example_labels.tar.gz

#conda activate ampseminer
python End_to_end_Tok_CLS/train_Tokens.py \
    --data_path ./example_data \
    --label_path ./example_labels \
    --model_name facebook/esm2_t30_150M_UR50D \
    --num_classes 2 \
    --ft_mode full \
    --epochs 10 \
    --batch_size 16 \
</pre>

## Custom code for Colab Notebook

## Datasource

## Citation

## Contacts

liwh@zhejianglab.org, liwh@tongji.edu.cn