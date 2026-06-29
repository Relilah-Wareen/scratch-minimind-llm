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
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings("ignore")


def dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """DPO loss：拉高 chosen 似然，压低 rejected 似然"""
    chosen_ratios = chosen_logps - ref_chosen_logps
    rejected_ratios = rejected_logps - ref_rejected_logps
    loss = -F.logsigmoid(beta * (chosen_ratios - rejected_ratios)).mean()
    chosen_rewards = beta * chosen_ratios.detach()
    rejected_rewards = beta * rejected_ratios.detach()
    return loss, chosen_rewards.mean(), rejected_rewards.mean()


def get_logps(model, input_ids, attention_mask, labels, loss_mask):
    """计算序列的对数概率"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = model.lm_head(outputs[0]) if isinstance(outputs, tuple) else model.lm_head(outputs)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs[:, :-1], -1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    masked_logps = token_logps * loss_mask[:, 1:]
    return masked_logps.sum(dim=-1) / (loss_mask[:, 1:].sum(dim=-1) + 1e-8)


def train_epoch(epoch, loader, iters, ref_model, start_step=0, wandb=None):
    start_time = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        x_c = batch["x_chosen"].to(args.device)
        y_c = batch["y_chosen"].to(args.device)
        mask_c = batch["mask_chosen"].to(args.device)
        attn_c = batch["attention_mask_chosen"].to(args.device)
        x_r = batch["x_rejected"].to(args.device)
        y_r = batch["y_rejected"].to(args.device)
        mask_r = batch["mask_rejected"].to(args.device)
        attn_r = batch["attention_mask_rejected"].to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            chosen_logps = get_logps(model, x_c, attn_c, y_c, mask_c)
            rejected_logps = get_logps(model, x_r, attn_r, y_r, mask_r)
            with torch.no_grad():
                ref_chosen_logps = get_logps(ref_model, x_c, attn_c, y_c, mask_c)
                ref_rejected_logps = get_logps(ref_model, x_r, attn_r, y_r, mask_r)
            loss, chosen_r, rejected_r = dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=args.dpo_beta
            )
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
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f"Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.4f} "
                f"chosen_r:{chosen_r.item():.3f} rejected_r:{rejected_r.item():.3f} lr:{current_lr:.8f} eta:{eta_min:.0f}min"
            )
            if wandb is not None:
                wandb.log({"loss": current_loss, "chosen_r": chosen_r.item(), "rejected_r": rejected_r.item(), "lr": current_lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step,
                wandb=wandb, save_dir="../checkpoints",
            )
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HelloModel DPO")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="dpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO KL 惩罚系数")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=2048, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO 数据路径")
    parser.add_argument("--from_weight", default="full_sft", type=str)
    parser.add_argument("--ref_weight", default="full_sft", type=str, help="参考模型权重")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="HelloModel-DPO")

    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = HelloModelConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"HelloModel-DPO-E{args.epochs}", id=wandb_id, resume=resume)

    # 训练模型（policy）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 参考模型（frozen，不更新）
    ref_model, _ = init_model(lm_config, args.ref_weight, device=args.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    Logger("Reference model frozen")

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

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
            Logger(f"Epoch[{epoch+1}/{args.epochs}]: skip {start_step} steps")
            train_epoch(epoch, loader, len(loader) + skip, ref_model, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()
