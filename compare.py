import torch, warnings
from transformers import AutoTokenizer
from model.model import HelloModelConfig, HelloModelForCausalLM

warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("./model")
config = HelloModelConfig()

prompts = [
    "你有什么特长？",
    "解释一下光合作用的基本过程",
    "请用Python写一个斐波那契数列的函数",
]

models = {
    "pretrain": "预训练",
    "full_sft": "SFT",
    "dpo": "DPO",
}

for weight, name in models.items():
    print(f"\n{'='*60}")
    print(f"  {name} 模型 ({weight})")
    print(f"{'='*60}")

    model = HelloModelForCausalLM(config)
    model.load_state_dict(torch.load(f"out/{weight}_512.pth", map_location="cpu"), strict=True)
    model.eval()

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=120, do_sample=True, temperature=0.85, top_p=0.85,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"\n👶 {prompt}")
        print(f"🤖 {reply[:300]}")
