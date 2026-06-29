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
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model import HelloModelConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model,
)

warnings.filterwarnings("ignore")


# ── 给 Actor 加 Value Head 构成 Actor-Critic ──
def add_value_head(model, hidden_size):
    model.value_head = nn.Linear(hidden_size, 1).to(next(model.parameters()).device)


def ppo_loss(log_probs, old_log_probs, advantages, values, returns, clip_eps=0.2, vf_coef=0.1):
    """PPO clipped loss + value loss"""
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    return policy_loss + vf_coef * value_loss


def get_advantages_and_returns(rewards, values, gamma=0.95, lam=0.95):
    """GAE 优势估计"""
    advantages, returns_list = [], []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns_list.insert(0, gae + values[t])
    return torch.tensor(advantages), torch.tensor(returns_list)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]
        answers = batch["answer"]

        # 1. 用旧策略生成回复
        responses, old_logps, values = generate_with_values(model, tokenizer, prompts)
        old_logps = old_logps.to(args.device)
        values = values.to(args.device)

        # 2. 计算简单奖励（长度适中 = 好）
        rewards = torch.tensor([
            min(len(tokenizer.decode(r, skip_special_tokens=True).strip()) / 200, 1.0)
            for r in responses
        ]).to(args.device)

        # 3. GAE
        advantages, returns = get_advantages_and_returns(rewards, values)
        advantages = advantages.to(args.device)
        returns = returns.to(args.device)

        # 4. 重新前向
        with autocast_ctx:
            logps_list, value_list = [], []
            for prompt, resp_ids in zip(prompts, responses):
                full_text = tokenizer.decode(
                    tokenizer(prompt, truncation=True).input_ids + resp_ids.tolist(),
                    skip_special_tokens=True,
                )
                full_ids = tokenizer(full_text, return_tensors="pt", truncation=True).to(args.device)
                out = model(input_ids=full_ids.input_ids, labels=full_ids.input_ids)
                logps_list.append(-out.loss * full_ids.input_ids.shape[1])
            log_probs = torch.stack(logps_list)

            loss = ppo_loss(log_probs, old_logps, advantages, values, returns, args.clip_eps)
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
                f"reward:{rewards.mean().item():.3f} eta:{eta_min:.0f}min"
            )
            if wandb is not None:
                wandb.log({"loss": current_loss, "reward": rewards.mean().item()})

        if step % args.save_interval == 0 and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw = model.module if isinstance(model, DistributedDataParallel) else model
            raw = getattr(raw, "_orig_mod", raw)
            torch.save({k: v.half().cpu() for k, v in raw.state_dict().items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir="../checkpoints")
            model.train()


@torch.no_grad()
def generate_with_values(model, tokenizer, prompts, max_new_tokens=256):
    """生成回复并返回 old_logps + values"""
    responses, logps_list, value_list = [], [], []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        resp_ids = out[0][inputs["input_ids"].shape[1]:]
        responses.append(resp_ids)
        # 用当前模型重算概率
        full_ids = tokenizer.decode(
            tokenizer(prompt, truncation=True).input_ids + resp_ids.tolist(),
            skip_special_tokens=True,
        )
        full_in = tokenizer(full_ids, return_tensors="pt", truncation=True).to(model.device)
        out2 = model(input_ids=full_in.input_ids, labels=full_in.input_ids)
        logps_list.append(-out2.loss * full_in.input_ids.shape[1])
        value_list.append(torch.zeros(1).to(model.device))
    return responses, torch.stack(logps_list), torch.stack(value_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HelloModel PPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="ppo_actor", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--clip_eps", type=float, default=0.2)
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
    parser.add_argument("--wandb_project", type=str, default="HelloModel-PPO")

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
        wandb.init(project=args.wandb_project, name=f"HelloModel-PPO", id=wandb_id, resume="must" if wandb_id else None)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    add_value_head(model, args.hidden_size)
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
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
