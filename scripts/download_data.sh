#!/bin/bash
# AutoDL 上一键下载所有训练数据
set -e

BASE="https://modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master"

echo "=== 下载 SFT 数据 ==="
wget -q --show-progress -O dataset/sft_t2t_mini.jsonl "$BASE&FilePath=sft_t2t_mini.jsonl"

echo "=== 下载 DPO 数据 ==="
wget -q --show-progress -O dataset/dpo.jsonl "$BASE&FilePath=dpo.jsonl"

echo "=== 下载 RLAIF 数据 ==="
wget -q --show-progress -O dataset/rlaif.jsonl "$BASE&FilePath=rlaif.jsonl"

echo "=== 全部下载完成 ==="
ls -lh dataset/*.jsonl
