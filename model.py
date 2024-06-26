import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig

class MultiHeadAttention(nn.Module):
    def __init__(self,
                    config: GPTConfig,
                    device="cpu",
                    dtype=torch.float32) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, device=device, dtype=dtype)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=config.n_embed,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
            device=device,
            dtype=dtype
        )
    
    def forward(self, x):
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0, 0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y

class FFN(nn.Module):
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.n_embed * 4, config.n_embed)
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x
    
class Block(nn.Module):
    def __init__(self, config:GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadAttention(config)
        self.mlp = FFN(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class EmbeddingStem(nn.Module):
    def __init__(self, config: GPTConfig, device="cpu", dtype=torch.float32) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed, device=device, dtype=dtype)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed, device=device, dtype=dtype))
        self.drop = nn.Dropout(config.embed_pdrop)
        self.block_size = config.block_size
    
    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        token_embedding = self.tok_emb(idx) # each index maps to a (learnable) embedding vector
        positional_embedding = self.pos_embed[:, :t, :] # each position maps to a (learnable) position vector
        return self.drop(token_embedding + positional_embedding)

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.block_size = config.block_size
        config = self._set_model_config(config)
        self.embed_stem = EmbeddingStem(config)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                p.data.normal_(mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        n_params = sum(p.numel() for p in self.blocks.parameters())
        n_params += sum(p.numel() for p in self.lm_head.parameters())
        print(n_params)
        print("number of parameters:%.2dM" % (n_params / 1e6,))
    
    def _set_model_config(self, config:GPTConfig):
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_embed is not None, config.n_head is not None])
        if type_given and not params_given:
            config.__dict__.update({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embed=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embed=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embed=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embed=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embed=48),
            }[config.model_type])
        return config
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self,idx, targets=None):
        x = self.embed_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for i in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # topk 采样，只选logits最高的topk个
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 归一化概率
            probs = F.softmax(logits, dim=-1)
            # 采样或者greedy search
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
