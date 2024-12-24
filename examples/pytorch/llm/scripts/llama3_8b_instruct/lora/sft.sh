# Experimental environment: 4 * A100
# 4 * 40GB GPU memory
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH
nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODELSCOPE_CACHE=/scratch/bcjw/boqiny2/ms-swift/cache \
swift sft \
    --model_type qwen2_5-7b-instruct \
    --model_revision master \
    --sft_type full \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output-1224 \
    --ddp_backend nccl \
    --dataset logic_reasoning_dataset.json \
    --train_dataset_sample -1 \
    --num_train_epochs 3 \
    --max_length 512 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 2 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn true \
    --deepspeed 'default-zero3' \
    --save_only_model true \
