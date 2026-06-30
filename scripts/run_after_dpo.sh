#!/bin/bash
set -e
cd trainer

echo "### [1/2] GRPO 策略优化 ###"
python train_grpo.py \
    --save_weight grpo \
    --epochs 1 --batch_size 2 --learning_rate 1e-6 \
    --data_path ../dataset/rlaif.jsonl \
    --num_generations 4 --beta 0.04 --reasoning 0 \
    --reward_model_path ../../internlm2-1_8b-reward
echo "GRPO 完成"

echo ""
echo "### [2/2] LoRA 低秩适配 ###"
python train_lora.py \
    --from_weight full_sft --lora_name lora_identity \
    --epochs 50 --batch_size 32 --learning_rate 1e-4 \
    --data_path ../dataset/sft_t2t_mini.jsonl
echo "LoRA 完成"

echo ""
echo "全部完成"
