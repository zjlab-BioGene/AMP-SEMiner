export CUDA_VISIBLE_DEVICES='0'

eval_script='eval_Tokens.CE_LoRA.py'
model_name='/home/lwh/00.data/AMP/train/esm_token_focalloss/finetune/esm2_t33_650M_UR50D-rank-8-ft-for-TokenCLS-labelSmth-0.0/epoch11-checkpoint-2045802'

apd_data='/home/lwh/00.data/AMP/dataset/labels/indenpendAPD/dataset'
apd_label='/home/lwh/00.data/AMP/dataset/labels/indenpendAPD/labels'

lamp2_data='/home/lwh/00.data/AMP/dataset/labels/indenpendLAMP2/dataset'
lamp2_label='/home/lwh/00.data/AMP/dataset/labels/indenpendLAMP2/labels'

## test-1
python $eval_script --outdir ./evaluation --model_name $model_name

## test_APD
python $eval_script --data_path $apd_data --label_path $apd_label --outdir ./APD_evaluation --model_name $model_name

## test_LAMP2
python $eval_script --data_path $lamp2_data --label_path $lamp2_label --outdir ./LAMP2_evaluation --model_name $model_name

