# Set the path to save checkpoints
OUTPUT_DIR='outputs/ssv2_linear_probe_sample3'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='list_ssv2'
# path to pretrain model
MODEL_PATH='checkpoints/vmae_vitb_ssv2_checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12320 --nnodes=2  --node_rank=$1 --master_addr=10.102.2.$2 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 48 \
    --num_sample 3 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --num_workers 16 \
#    --enable_deepspeed