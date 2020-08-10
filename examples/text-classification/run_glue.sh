export GLUE_DIR=/media/drive/Datasets/glue_data
export TASK_NAME=MRPC

export OUTPUT_DIR_NAME=mrpc_bert_mms_loss2
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

CUDA_VISIBLE_DEVICES=0 python ./examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $OUTPUT_DIR






#python ./examples/text-classification/run_glue.py --model_name_or_path bert-base-uncased --task_name MRPC --do_train --do_eval --data_dir /media/drive/Datasets/glue_data/MRPC
#    --max_seq_length 128
#    --per_device_eval_batch_size=8
#    --per_device_train_batch_size=8
#    --learning_rate 2e-5
#    --num_train_epochs 3.0
#    --output_dir ./test