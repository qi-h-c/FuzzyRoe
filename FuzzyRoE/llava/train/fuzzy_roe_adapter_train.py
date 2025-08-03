import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import jsonlines
import llava
from torch.cuda.amp import autocast


class Fuzzy_Router(nn.Module):
    def __init__(self, in_features=768):
        super().__init__()
        self.in_features = in_features
        self.temp = 1.0  
        
        self.fuzzy_rules = [
            # (0,0,0)
            {"antecedent": {"x1": (0.00, 0.33), "x2": (0.00, 0.33), "x3": (0.00, 0.33)}, "consequent": 0.000},
            # (0,0,2)
            {"antecedent": {"x1": (0.00, 0.33), "x2": (0.00, 0.33), "x3": (0.67, 1.00)}, "consequent": 0.333},
            # (0,1,1)
            {"antecedent": {"x1": (0.00, 0.33), "x2": (0.33, 0.67), "x3": (0.33, 0.67)}, "consequent": 0.333},
            # (0,2,0)
            {"antecedent": {"x1": (0.00, 0.33), "x2": (0.67, 1.00), "x3": (0.00, 0.33)}, "consequent": 0.333},
            # (0,2,2)
            {"antecedent": {"x1": (0.00, 0.33), "x2": (0.67, 1.00), "x3": (0.67, 1.00)}, "consequent": 0.667},
            # (1,0,0)
            {"antecedent": {"x1": (0.33, 0.67), "x2": (0.00, 0.33), "x3": (0.00, 0.33)}, "consequent": 0.333},
            # (1,0,2)
            {"antecedent": {"x1": (0.33, 0.67), "x2": (0.00, 0.33), "x3": (0.67, 1.00)}, "consequent": 0.667},
            # (1,1,1)
            {"antecedent": {"x1": (0.33, 0.67), "x2": (0.33, 0.67), "x3": (0.33, 0.67)}, "consequent": 0.500},
            # (1,2,0)
            {"antecedent": {"x1": (0.33, 0.67), "x2": (0.67, 1.00), "x3": (0.00, 0.33)}, "consequent": 0.667},
            # (1,2,2)
            {"antecedent": {"x1": (0.33, 0.67), "x2": (0.67, 1.00), "x3": (0.67, 1.00)}, "consequent": 1.000},
            # (2,0,0)
            {"antecedent": {"x1": (0.67, 1.00), "x2": (0.00, 0.33), "x3": (0.00, 0.33)}, "consequent": 0.667},
            # (2,0,2)
            {"antecedent": {"x1": (0.67, 1.00), "x2": (0.00, 0.33), "x3": (0.67, 1.00)}, "consequent": 1.000},
            # (2,1,1)
            {"antecedent": {"x1": (0.67, 1.00), "x2": (0.33, 0.67), "x3": (0.33, 0.67)}, "consequent": 1.000},
            # (2,2,0)
            {"antecedent": {"x1": (0.67, 1.00), "x2": (0.67, 1.00), "x3": (0.00, 0.33)}, "consequent": 1.167},
            # (2,2,2)
            {"antecedent": {"x1": (0.67, 1.00), "x2": (0.67, 1.00), "x3": (0.67, 1.00)}, "consequent": 1.500},
        ]
        
    def compute_fuzzy_set_Rep(self, feature: torch.Tensor, mode="entropy") -> torch.Tensor:
        if mode == "entropy":
            attn_sim = torch.softmax(feature, dim=-1)             # [B, T, D]
            entropy = - (attn_sim * torch.log(attn_sim + 1e-8)).sum(dim=-1)  # [B, T]
            return entropy / entropy.max().clamp(min=1e-6)    
        elif mode == "variance":
            return feature.var(dim=-1) / 10.0                     # [B, T]  
        elif mode == "mean":
            return feature.mean(dim=-1)                           # [B, T] 
        else:
            return torch.full((feature.shape[0], feature.shape[1]), 0.5, device=feature.device)

    def match_rule(self, x1: float, x2: float, x3: float) -> Optional[float]:
        for rule in self.fuzzy_rules:
            x1_range = rule["antecedent"]["x1"]
            x2_range = rule["antecedent"]["x2"]
            x3_range = rule["antecedent"]["x3"]
            if (x1_range[0] <= x1 < x1_range[1] and
                x2_range[0] <= x2 < x2_range[1] and
                x3_range[0] <= x3 < x3_range[1]):
                return rule["consequent"]
        return None  
    def interpolate_rule(self, x1: float, x2: float, x3: float) -> float:
        distances = []
        for rule in self.fuzzy_rules:
            c1 = (rule["antecedent"]["x1"][0] + rule["antecedent"]["x1"][1]) / 2
            c2 = (rule["antecedent"]["x2"][0] + rule["antecedent"]["x2"][1]) / 2
            c3 = (rule["antecedent"]["x3"][0] + rule["antecedent"]["x3"][1]) / 2
            dist = ((x1 - c1)**2 + (x2 - c2)** 2 + (x3 - c3)**2) **0.5
            distances.append((dist, rule["consequent"]))
        
        distances.sort(key=lambda x: x[0])
        d1, f1 = distances[0]
        d2, f2 = distances[1]
        lambda_ = d1 / (d1 + d2) if (d1 + d2) != 0 else 0.5
        return (1 - lambda_) * f1 + lambda_ * f2

    def forward(self, x: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        # 提取多模态特征
        image_tokens = x[:, 0:1, :]
        text_tokens = x[:, 1:question_mask.shape[1], :]
        
        x1 = self.compute_fuzzy_set_Rep(image_tokens, mode="entropy")     # [B, 1]
        x2 = self.compute_fuzzy_set_Rep(text_tokens, mode="entropy")      # [B, T_text]
        x3 = F.cosine_similarity(image_tokens.mean(dim=1), text_tokens.mean(dim=1))  # [B]
        x3 = x3.unsqueeze(1).expand(-1, text_tokens.size(1))               # [B, T_text]

        # 拼接成 [B, T_text, 3]
        feature_rep = torch.stack([x1.expand_as(x2), x2, x3], dim=-1)      # [B, T_text, 3]

        f_list = []
        for i in range(feature_rep.size(0)):      # 遍历 batch
            f_token = []
            for t in range(feature_rep.size(1)):  # 遍历 token
                x1, x2, x3 = feature_rep[i, t].tolist()
                f_val = self.match_rule(x1, x2, x3)
                if f_val is None:
                    f_val = self.interpolate_rule(x1, x2, x3)
                f_token.append(f_val)
            f_list.append(f_token)

        f_tensor = torch.tensor(f_list, device=x.device)  # [B, T_text]
        batch_size, seq_len = x.shape[:2]
        weight = torch.zeros(batch_size, seq_len, 2, device=x.device)
        weight[:, 1:1+f_tensor.shape[1], 0] = f_tensor      # 主路径
        weight[:, 1:1+f_tensor.shape[1], 1] = 1 - f_tensor  # adapter路径
        
        # 应用温度参数并softmax
        #return torch.softmax(weight / self.temp, dim=-1)
        return weight

class Adapter(nn.Module):
    def __init__(
        self,
        in_features=768,
        hidden_dim=8,
        scale=1,
    ):
        super().__init__()
        self.conv_A = nn.Linear(in_features, hidden_dim, bias=False)
        self.conv_B = nn.Linear(hidden_dim, in_features, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.xavier_uniform_(self.conv_B.weight)
        
    def forward(self, x):
        x = self.conv_A(x)
        x = self.act(x)
        x = self.conv_B(x)
        return x


def forward_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        question_mask: Optional[torch.Tensor] = None,
        weight_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # 正常 layernorm
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # 得到 token 级路由权重（f）
    # weight = self.adapter_MHA_router(hidden_states, question_mask)  # [B, T, 2]
    # w_main = weight[:, :, 0].unsqueeze(-1)  # [B, T, 1]
    # w_skip = weight[:, :, 1].unsqueeze(-1)  # [B, T, 1]
    if (question_mask is not None):
        self.weight = self.adapter_MHA_router(hidden_states, question_mask) # B x T x 2
    else:
        self.weight = True
    w_main = (self.weight[:, :, 0, None] * weight_mask).sum(1) # B x N
    w_skip = (self.weight[:, :, 1, None] * weight_mask).sum(1) # B x N

    # 1. 计算主路径输出
    hidden_states = hidden_states.to(next(self.parameters()).dtype)

    hidden_main, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
    hidden_main = residual + hidden_main
    res_main = hidden_main
    hidden_main = hidden_main.to(next(self.parameters()).dtype)
    res_main = res_main.to(next(self.parameters()).dtype)
    hidden_main = self.post_attention_layernorm(hidden_main)
    hidden_main = self.mlp(hidden_main)
    hidden_main = res_main + hidden_main  # [B, T, D]

    # 2. 计算跳跃路径（Adapter）输出
    hidden_skip = residual + self.adapter_MHA(hidden_states)  # [B, T, D]

    # 3. 按 token 权重融合输出
    hidden_states = w_main * hidden_main + w_skip * hidden_skip  # [B, T, D]

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    if self.training:
        output_weight = self.weight #* question_mask
        output_weight = output_weight[:, 1:, 1].sum(1)
        outputs += (output_weight, )
        #print(f'skip_ratio_layer: {skip_ratio_layer}')
    return outputs


def set_Fuzzy_Adapter(model, dim=8, s=1, set_forward=True, t=1, gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == llava.model.language_model.modeling_llama.LlamaDecoderLayer:
            # 替换为模糊路由
            _.adapter_MHA = Adapter(_.hidden_size, 1024).half()
            _.adapter_MHA_router = Fuzzy_Router(_.hidden_size).half()  # 关键：使用Fuzzy_Router
            _.weight = None
            _.s = s
            _.t = t
            _.skip_flag = None  # 新增跳过标记
            
            # 绑定修改后的前向传播
            bound_method = forward_llama.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_Fuzzy_Adapter(_, dim, s, set_forward=set_forward, t=t, gradient_checkpointing=gradient_checkpointing)
