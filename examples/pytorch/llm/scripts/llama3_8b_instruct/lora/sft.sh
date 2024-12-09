# Experimental environment: 4 * A100
# 4 * 40GB GPU memory
nproc_per_node=8

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model_type llama3-8b-instruct \
    --model_revision master \
    --sft_type full \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output-1203 \
    --ddp_backend nccl \
    --dataset logic_reasoning_dataset.json \
    --train_dataset_sample -1 \
    --num_train_epochs 3 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 3 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn true \
    --deepspeed 'default-zero3' \
    --save_only_model true \
