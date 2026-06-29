from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── 全局工具函数 ──

SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model.",
]


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """以一定概率随机插入 system 消息，增加模型对有无 system prompt 的泛化能力"""
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """清理 chat template 渲染后可能出现的空 <think> 块"""
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


# ── 1. PretrainDataset —— 自回归预训练数据集 ──
# 目标：Next-Token Prediction
# 格式：{"text": "一段原始文本"}
# labels 直接 clone 自 input_ids，PAD 位置置 -100 不参与 loss

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index): 
        # 拿到的是jsonl的每一行
        # tokenizer把文本转化为input_id
        # 加上EOS,BOS,PAD
        # 自行编写labels,防止PAD参与loss计算
        # 输出input_ids, attention_mask, labels
        sample = self.samples[index]
        # tokenize，预留 BOS + EOS 的位置
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        # 拼接 BOS + token序列 + EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 右侧 PAD 补齐
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask


# ── 2. SFTDataset —— 有监督微调数据集 ──
# 目标：只预测 assistant 回复，user/system 部分不参与 loss
# 格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 预 tokenize assistant 回复的起止标记，用于定位回复区间
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """将多轮对话渲染为模型输入字符串"""
        messages = []
        for msg in conversations:
            msg = dict(msg)
            msg.pop("functions", None)     # 去掉 function calling
            msg.pop("tool_calls", None)    # 去掉 tool calling，防止 Jinja tojson 报错
            messages.append(msg)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def generate_labels(self, input_ids):
        """滑动窗口扫描 bos_id，只有 assistant 回复区间有有效 label，其余为 -100"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample["conversations"])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)

        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        attention_mask = (
            torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_token_id
        ).long()
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            attention_mask,
        )


# ── 3. DPODataset —— 直接偏好优化数据集 ──
# 目标：最大化 chosen 似然、最小化 rejected 似然
# 格式：{"chosen": [...], "rejected": [...]}

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def generate_loss_mask(self, input_ids):
        """生成 0/1 loss mask，只有 assistant 回复区间为 1"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                chosen, tokenize=False, add_generation_prompt=False
            )
        )
        rejected_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(
                rejected, tokenize=False, add_generation_prompt=False
            )
        )

        chosen_enc = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )
        rejected_enc = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length"
        )

        c_ids = chosen_enc["input_ids"]
        c_mask = self.generate_loss_mask(c_ids)
        r_ids = rejected_enc["input_ids"]
        r_mask = self.generate_loss_mask(r_ids)

        return {
            "x_chosen": torch.tensor(c_ids[:-1], dtype=torch.long),
            "y_chosen": torch.tensor(c_ids[1:], dtype=torch.long),
            "mask_chosen": torch.tensor(c_mask[1:], dtype=torch.long),
            "x_rejected": torch.tensor(r_ids[:-1], dtype=torch.long),
            "y_rejected": torch.tensor(r_ids[1:], dtype=torch.long),
            "mask_rejected": torch.tensor(r_mask[1:], dtype=torch.long),
            "attention_mask_chosen": (torch.tensor(c_ids[:-1]) != self.padding).long(),
            "attention_mask_rejected": (torch.tensor(r_ids[:-1]) != self.padding).long(),
        }