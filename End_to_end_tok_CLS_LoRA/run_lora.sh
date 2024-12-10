export CUDA_VISIBLE_DEVICES=0

#epoch 1-12 > run_lora.sh.log2
#python train_Tokens_FOCALloss_labelSmt.py --data_path ./data --epochs 12 --batch_size 8 --model_name ../../basemodels/esm2_t33_650M_UR50D --focal 0.0  --lr 1e-4 --ft_mode lora --lora_rank 8

#epoch 13-20 > run_lora.sh.log3
#python train_Tokens_FOCALloss_labelSmt.py --data_path ./data --epochs 20 --batch_size 8 --model_name ../../basemodels/esm2_t33_650M_UR50D --focal 0.0  --lr 1.2725245812318756e-8 --ft_mode lora --lora_rank 8

#epoch 21-32 > run_lora.sh.log4
python train_Tokens_FOCALloss_labelSmt.py --data_path ./data --epochs 32 --batch_size 8 --model_name ../../basemodels/esm2_t33_650M_UR50D --focal 0.0  --lr 1.2725245812318756e-8 --ft_mode lora --lora_rank 8
