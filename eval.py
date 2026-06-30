import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import HelloModelConfig, HelloModelForCausalLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings("ignore")


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = HelloModelForCausalLM(
        HelloModelConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
        )
    )
    moe_suffix = "_moe" if args.use_moe else ""
    ckp = f"out/{args.weight}_{args.hidden_size}{moe_suffix}.pth"
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="HelloModel 推理")
    parser.add_argument("--load_from", default="model", type=str, help="tokenizer 路径")
    parser.add_argument("--weight", default="full_sft", type=str, help="权重名称前缀")
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--inference_rope_scaling", default=False, action="store_true")
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.85, type=float)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--historys", default=0, type=int, help="携带历史轮数")
    parser.add_argument("--mode", default="auto", type=str, choices=["auto", "manual"], help="auto=自动测试, manual=手动输入")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用 Python 写一个计算斐波那契数列的函数",
        "解释一下光合作用的基本过程",
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = 0 if args.mode == "auto" else 1
    if input_mode == 1:
        print("手动输入模式，输入 quit 退出")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("👶: "), "")
    for prompt in prompt_iter:
        setup_seed(42)
        if input_mode == 0:
            print(f"👶: {prompt}")
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        text = tokenizer.apply_chat_template(**templates) if args.weight != "pretrain" else (tokenizer.bos_token + prompt)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(args.device)

        print("🤖: ", end="")
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0,
        )
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": response})
        print()


if __name__ == "__main__":
    main()
