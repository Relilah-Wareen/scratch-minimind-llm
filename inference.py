import torch
from model.model import HelloModelConfig, HelloModelForCausalLM
from transformers import AutoTokenizer

# 1. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("./model")
config = HelloModelConfig()
model = HelloModelForCausalLM(config)

# 2. 加载预训练权重
weights = torch.load("out/pretrain_512.pth", map_location="cpu")
model.load_state_dict(weights, strict=False)
model.eval()
print("模型加载完成")

# 3. 生成
prompt = "人工智能的发展"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))