export CUDA_VISIBLE_DEVICES='0'

eval_script='eval_2steps.py'
model_name1='finetune/esm2_t33_650M_UR50D-ft-for-sequence-classification-full-finetune/epoch20-checkpoint-1859820'
#model_name2='/home/lwh/00.data/AMP/trainedmodels/ftTokenCLS/esm2_t33_650M_UR50D-full-ft-for-Token-classification/epoch15-checkpoint-1394865'
model_name2='finetune/esm2_t33_650M_UR50D-ft-for-Token-classification-full-finetune/epoch50-checkpoint-41550'

# apd_data='/home/lwh/00.data/AMP/dataset/labels/indenpendAPD/dataset'
# apd_label='/home/lwh/00.data/AMP/dataset/labels/indenpendAPD/labels'

# lamp2_data='/home/lwh/00.data/AMP/dataset/labels/indenpendLAMP2/dataset'
# lamp2_label='/home/lwh/00.data/AMP/dataset/labels/indenpendLAMP2/labels'

## test-1
# python $eval_script --outdir ./evaluation_lwh --model_name1 $model_name1 --model_name2 $model_name2

## test_APD
# python $eval_script --data_path $apd_data --label_path $apd_label --outdir ./APD_evaluation --model_name1 $model_name1 --model_name2 $model_name2

## test_LAMP2
# python $eval_script --data_path $lamp2_data --label_path $lamp2_label --outdir ./LAMP2_evaluation --model_name1 $model_name1 --model_name2 $model_name2

python $eval_script --datatype val --outdir ./evaluation_val_lwh --model_name1 $model_name1 --model_name2 $model_name2