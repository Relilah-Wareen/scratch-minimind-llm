import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import HelloModelConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model,
)

warnings.filterwarnings("ignore")


# ── GRPO Loss ──
def grpo_loss(log_probs, old_log_probs, advantages, clip_eps=0.2, kl_coef=0.02):
    """Group Relative Policy Optimization 核心损失"""
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl = (old_log_probs - log_probs).mean()
    return policy_loss + kl_coef * kl, kl.detach()


def get_advantages(rewards):
    """从奖励计算优势：标准化到零均值"""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, max_new_tokens=256):
    """用当前模型批量生成回复并返回 token IDs + log_probs"""
    responses, log_probs_list = [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            output_logits=True, return_dict_in_generate=True,
        )
        ids = outputs.sequences[0]
        response_ids = ids[inputs["input_ids"].shape[1]:]
        responses.append(response_ids)
        logits = torch.stack(outputs.logits, dim=0).squeeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = torch.gather(log_probs, -1, response_ids.unsqueeze(-1)).squeeze(-1)
        log_probs_list.append(token_logps.sum())
    return responses, torch.stack(log_probs_list)


@torch.no_grad()
def reward_score(prompts, responses, tokenizer):
    """简单规则奖励：回复长度适中 + 不含重复 n-gram"""
    rewards = []
    for prompt, resp_ids in zip(prompts, responses):
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        score = 0.0
        length = len(text.strip())
        if 10 < length < 500:
            score += 1.0
        elif length > 0:
            score += 0.3
        # 惩罚重复
        words = text.split()
        if len(words) > 5 and len(set(words)) / len(words) < 0.5:
            score -= 0.5
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        answers = batch["answer"]

        # 1. 生成回复
        responses, old_logps = generate_responses(model, tokenizer, prompts, max_new_tokens=args.max_gen_len)

        # 2. 计算奖励 & 优势
        rewards = reward_score(prompts, responses, tokenizer).to(args.device)
        advantages = get_advantages(rewards).to(args.device)

        # 3. 前向 + loss（只更新模型，不更新ref）
        with autocast_ctx:
            # 重新计算当前模型对生成回复的 log_prob
            logps_list = []
            for prompt, resp_ids in zip(prompts, responses):
                full_text = tokenizer.decode(
                    tokenizer(prompt, truncation=True).input_ids + resp_ids.tolist(),
                    skip_special_tokens=True,
                )
                full_ids = tokenizer(full_text, return_tensors="pt", truncation=True).to(args.device)
                out = model(input_ids=full_ids.input_ids, labels=full_ids.input_ids)
                logps_list.append(-out.loss * full_ids.input_ids.shape[1])
            log_probs = torch.stack(logps_list).to(args.device)

            loss, kl = grpo_loss(log_probs, old_logps.to(args.device), advantages, args.clip_eps, args.kl_coef)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.4f} "
                f"reward:{rewards.mean().item():.3f} kl:{kl.item():.4f} eta:{eta_min:.0f}min"
            )
            if wandb is not None:
                wandb.log({"loss": current_loss, "reward": rewards.mean().item(), "kl": kl.item()})

        if step % args.save_interval == 0 and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw = model.module if isinstance(model, DistributedDataParallel) else model
            raw = getattr(raw, "_orig_mod", raw)
            torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir="../checkpoints")
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HelloModel GRPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="grpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--max_gen_len", default=256, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl")
    parser.add_argument("--from_weight", default="full_sft", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="HelloModel-GRPO")

    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = HelloModelConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe),
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        wandb.init(project=args.wandb_project, name=f"HelloModel-GRPO-E{args.epochs}", id=wandb_id, resume="must" if wandb_id else None)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
