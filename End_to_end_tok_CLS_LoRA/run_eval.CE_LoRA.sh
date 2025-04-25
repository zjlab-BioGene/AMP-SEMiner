export CUDA_VISIBLE_DEVICES='0'

eval_script='eval_Tokens.CE_LoRA.py'
model_name='YOUR_MODEL_PATH'

apd_data='YOUR_APD_DATASET_FOLDER'
apd_label='YOUR_APD_LABEL_FOLDER'

lamp2_data='YOUR_LAMP2_DATASET_FOLDER'
lamp2_label='YOUR_LAMP2_LABEL_FOLDER'

## test-1
python $eval_script --outdir ./evaluation --model_name $model_name

## test_APD
python $eval_script --data_path $apd_data --label_path $apd_label --outdir ./APD_evaluation --model_name $model_name

## test_LAMP2
python $eval_script --data_path $lamp2_data --label_path $lamp2_label --outdir ./LAMP2_evaluation --model_name $model_name

