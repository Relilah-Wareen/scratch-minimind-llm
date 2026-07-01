# HelloModel — 从零手写轻量 LLM

基于 MiniMind 教程，用 PyTorch 从数学推导到工程实现完整搭建了一个 **26M 参数**的 Decoder-Only 语言模型，跑通了从预训练到 RLHF 的完整训练链路。模型结构对齐 Qwen3 生态。

## 做了什么

- **模型架构**：逐模块手写了 RMSNorm、RoPE/YaRN 旋转位置编码、GQA 分组查询注意力、SwiGLU 门控前馈网络
- **训练管线**：实现 BPE Tokenizer 加载、数据集构建（Pretrain/SFT/DPO/RLAIF）、混合精度 + 分布式训练脚本
- **全流程跑通**：
  - **预训练**（127 万条文本，4090 ~1.5h）
  - **SFT 微调**（90 万条对话，学会问答格式）
  - **DPO 偏好对齐**（1.7 万条偏好对，改善回复质量）
  - **GRPO 策略优化**（1.9 万条 RLAIF，规则奖励强化）
  - **LoRA 身份注入**（低秩适配，几分钟注入身份记忆）

## 项目结构

```
├── model/
│   ├── model.py             # 模型架构（Config → RMSNorm → RoPE → Attention → FFN → Block → CausalLM）
│   └── model_lora.py       # LoRA 适配器
├── dataset/
│   └── lm_dataset.py       # 数据集（Pretrain / SFT / DPO / RLAIF）
├── trainer/
│   ├── train_pretrain.py   # 预训练
│   ├── train_full_sft.py   # SFT 微调
│   ├── train_dpo.py        # DPO 偏好对齐
│   ├── train_grpo.py       # GRPO 策略优化
│   ├── train_lora.py       # LoRA 低秩适配
│   └── trainer_utils.py    # 工具函数
├── scripts/
│   ├── download_data.sh    # 下载训练数据
│   └── run_all.sh          # 一键训练
├── eval.py                 # 推理对话
└── compare.py              # 多模型对比
```

## 快速开始

```bash
# 推理对话
python eval.py --weight full_sft

# 三模型对比
python compare.py
```

## 致谢

基于 [MiniMind](https://github.com/jingyaogong/minimind) 和 [MokioMind](https://github.com/Wood-Q/MokioMind) 项目学习实现。
