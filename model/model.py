from transformers import PretrainedConfig

class HelloModelConfig(PretrainedConfig):
    model_type = "hello_model"

    def __init__(
        self,
        dropout: float = 0.0,                # 训练时随机丢弃神经元比例，0=不丢弃
        bos_token_id: int = 1,               # 句子起始标记ID
        eos_token_id: int = 2,               # 句子结束标记ID
        hidden_act: str = "silu",            # FFN 激活函数类型 (silu/gelu)
        hidden_size: int = 512,              # 隐藏层维度 d_model，整个模型的向量宽度
        intermediate_size: int = None,       # FFN 中间升维大小，None 则自动按 8/3 倍计算
        max_position_embeddings: int = 32768,# 最大位置编码长度（推理时支持的最长 token 数）
        num_attention_heads: int = 8,        # Q 注意力头数（每个 head 独立关注不同子空间）
        num_hidden_layers: int = 8,          # Transformer Block 层数（深度）
        num_key_value_heads: int = 2,        # KV 头数（GQA：≤ Q 头数，省显存）
        vocab_size: int = 6400,              # 词表大小
        rms_norm_eps: float = 1e-05,         # RMSNorm 的 epsilon，防除零
        rope_theta: int = 1000000,           # RoPE 基础频率，越大低频越多，长距离外推越好
        inference_rope_scaling: bool = False,# 推理时是否启用 YaRN 长度外推
        flash_attention: bool = True,        # 是否使用 Flash Attention 加速
        ############ MoE ############
        use_moe: bool = False,               # 是否启用混合专家层（MoE）
        num_experts_per_tok: int = 2,        # 每个 token 激活的专家数 (top-k)
        n_routed_experts: int = 4,           # 可路由专家总数
        n_shared_experts: int = 1,           # 共享专家数（所有 token 都会经过）
        scoring_func: str = "softmax",       # 专家选择评分函数
        aux_loss_alpha: float = 0.01,        # 辅助负载均衡损失的权重
        seq_aux: bool = True,                # 辅助损失是否在序列维度计算
        norm_topk_prob: bool = True,         # 是否对 top-k 权重做归一化
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch 
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
# RMSnorm method 
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
# forward 
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)
    
def precompute_freqs_cis(dim:int,end:int(32*1024),rope_base,rope_scaling:Optional[dict]=None):
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )
        
    if rope_scaling is not None:
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        
        orig_max,factor,beta_fast,beta_slow,attn_factor=(
          rope_scaling["original_max_position_embeddings"],
          rope_scaling["factor"],
          rope_scaling["beta_fast"],
          rope_scaling["beta_slow"],
          rope_scaling["attention_factor"],
        )
        
        if end>orig_max:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            low,high=(max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2-1))

            # 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            freqs=freqs*(1-ramp+ramp/factor)

    t = torch.arange(end, device=freqs.device).float()

    # 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # 向量化实现 2D 旋转 (x cosθ - y sinθ, y cosθ + x sinθ)；全部 element-wise + cat，避免跳读和中间张量

        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )
        # (x cos θ - y sin θ, y cos θ + x sin θ)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep==1:
        return x
    
    return x[:,:,:,None,:].expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim)

class Attention(nn.Module):
    def __init__(self,args:HelloModelConfig):
        super().__init__()


        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 拆分多头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 对QK用roPE
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
    

class FeedForward(nn.Module):
    # 初始化

    # 升维

    # 降维

    # 门控

    # dropout

    # 激活函数

    def __init__(self, args:HelloModelConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size=int(args.hidden_size*8/3)  # 2.66 经验值
            args.intermediate_size=64*((intermediate_size+64-1)//64)
        
        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.dropout=nn.Dropout(args.dropout)
        self.act_fn=ACT2FN[args.hidden_act]
    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))

class HelloModelBlock(nn.Module):
    # Pre-Norm 架构：RMSNorm → Attention → +残差 → RMSNorm → FFN → +残差
    def __init__(self, layer_id: int, args: HelloModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)      # Attention 前的归一化
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)  # FFN 前的归一化
        self.self_attention = Attention(args)
        self.mlp = FeedForward(args)  # 后续可替换为 MoEFeedForward

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Pre-Norm Attention + 残差
        res = hidden_states
        hidden_states, present_kv = self.self_attention(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = res + hidden_states

        # Pre-Norm FFN + 残差
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )

        return hidden_states, present_kv
    
class HelloModel(nn.Module):
    # 完整 Decoder 模型：Embedding → N×Block → RMSNorm
    def __init__(self, args: HelloModelConfig):
        super().__init__()
        self.config = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [HelloModelBlock(l, args) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # 预计算 RoPE cos/sin 表并注册为 buffer（不参与训练，自动跟随 module 移动设备）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=args.hidden_size // args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 按当前 batch 的位置范围切出对应的 cos/sin
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        aux_loss = hidden_states.new_zeros(1).squeeze()
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )   
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents, aux_loss


class HelloModelForCausalLM(PreTrainedModel, GenerationMixin):
    # HuggingFace 因果语言模型包装：HelloModel + lm_head + loss
    config_class = HelloModelConfig

    def __init__(self, args: HelloModelConfig):
        super().__init__(args)
        self.model = HelloModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        # 共享权重：embedding 和 lm_head 用同一个矩阵（省参数 + 训练更稳）
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        # 推理时可能只需要最后几个 token 的 logits（省计算）
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 因果语言模型的标准 loss：预测第 n+1 个 token，所以移位一行
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output