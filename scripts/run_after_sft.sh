#!/bin/bash
set -e
cd trainer

echo "### [1/3] DPO 偏好对齐 ###"
python train_dpo.py \
    --from_weight full_sft --ref_weight full_sft --save_weight dpo \
    --epochs 1 --batch_size 1 --learning_rate 5e-7 \
    --data_path ../dataset/dpo.jsonl
echo "DPO 完成"

echo ""
echo "### [2/3] GRPO 策略优化 ###"
python train_grpo.py \
    --from_weight full_sft --save_weight grpo \
    --epochs 1 --batch_size 2 --learning_rate 1e-6 \
    --data_path ../dataset/rlaif.jsonl
echo "GRPO 完成"

echo ""
echo "### [3/3] LoRA 低秩适配 ###"
python train_lora.py \
    --from_weight full_sft --lora_name lora_identity \
    --epochs 50 --batch_size 32 --learning_rate 1e-4 \
    --data_path ../dataset/sft_t2t_mini.jsonl
echo "LoRA 完成"

echo ""
echo "全部完成"
