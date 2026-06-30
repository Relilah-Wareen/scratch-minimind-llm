import warnings
import torch
from transformers import AutoTokenizer, TextStreamer
from model.model import HelloModelConfig, HelloModelForCausalLM

warnings.filterwarnings("ignore")

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("./model")
config = HelloModelConfig()
model = HelloModelForCausalLM(config)

weights = torch.load("out/full_sft_512.pth", map_location="cpu")
model.load_state_dict(weights, strict=True)
model.eval()

print(f"HelloModel 已加载 (25.8M 参数)")
print("输入对话，输入 quit 退出\n")

conversation = []
while True:
    user = input("👶: ").strip()
    if user.lower() == "quit":
        break
    if not user:
        continue

    conversation.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    print("🤖: ", end="", flush=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generated = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.85,
        top_p=0.85,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    reply = tokenizer.decode(
        generated[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
    )
    conversation.append({"role": "assistant", "content": reply})
    print()
