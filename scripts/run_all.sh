#!/bin/bash
# AutoDL 顺序执行全部训练：SFT → DPO → GRPO → LoRA
# 使用：bash scripts/run_all.sh
set -e

echo "========================================="
echo " HelloModel 全流程训练"
echo " 顺序: SFT → DPO → GRPO → LoRA"
echo "========================================="

cd trainer

# ── 1. SFT ──
echo ""
echo "### [1/4] SFT 有监督微调 ###"
python train_full_sft.py \
    --from_weight pretrain --save_weight full_sft \
    --epochs 2 --batch_size 16 --learning_rate 1e-6 \
    --data_path ../dataset/sft_t2t_mini.jsonl

echo "SFT 完成！权重: ../out/full_sft_512.pth"

# ── 2. DPO ──
echo ""
echo "### [2/4] DPO 偏好对齐 ###"
python train_dpo.py \
    --from_weight full_sft --ref_weight full_sft --save_weight dpo \
    --epochs 1 --batch_size 1 --learning_rate 5e-7 \
    --data_path ../dataset/dpo.jsonl

echo "DPO 完成！权重: ../out/dpo_512.pth"

# ── 3. GRPO ──
echo ""
echo "### [3/4] GRPO 策略优化 ###"
python train_grpo.py \
    --from_weight full_sft --save_weight grpo \
    --epochs 1 --batch_size 2 --learning_rate 1e-6 \
    --data_path ../dataset/rlaif.jsonl

echo "GRPO 完成！权重: ../out/grpo_512.pth"

# ── 4. LoRA ──
echo ""
echo "### [4/4] LoRA 低秩适配 ###"
python train_lora.py \
    --from_weight full_sft --lora_name lora_identity \
    --epochs 50 --batch_size 32 --learning_rate 1e-4 \
    --data_path ../dataset/sft_t2t_mini.jsonl

echo ""
echo "========================================="
echo " 全部训练完成！"
echo "========================================="
