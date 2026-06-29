import torch
from torch import nn


class LoRA(nn.Module):
    """低秩适配器：W' = W + B @ A，其中 A 和 B 是低秩矩阵"""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维 (in → rank)
        self.B = nn.Linear(rank, out_features, bias=False)  # 升维 (rank → out)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """给模型中所有方阵 Linear 层附加 LoRA 适配器"""
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(device)
            setattr(module, "lora", lora)
            original_forward = module.forward
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora


def load_lora(model, path):
    """加载已保存的 LoRA 权重"""
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = name + ".lora"
            lora_state = {k.replace(prefix + ".", ""): v for k, v in state_dict.items() if k.startswith(prefix)}
            if lora_state:
                module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """只保存 LoRA 部分的权重"""
    lora_state = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            for k, v in module.lora.state_dict().items():
                lora_state[f"{name}.lora.{k}"] = v
    torch.save(lora_state, path)
