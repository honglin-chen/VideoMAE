# Set the path to save checkpoints
OUTPUT_DIR='outputs/cwm_kinetics_ssv2_finetune_0514'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='list_ssv2'
# path to pretrain model
MODEL_PATH='/ccn2/u/honglinc/cwm_checkpoints/ablation_3frame_no_clumping_mr0.90_extra_data_ep400/checkpoint-399.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12321 --nnodes=4  --node_rank=$1 --master_addr=10.102.2.$2 \
    run_class_finetuning.py \
    --model vit_base_patch16_224_cwm \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --update_freq 2 \
    --enable_deepspeed

