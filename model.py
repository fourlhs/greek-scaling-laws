from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 16000
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        Q, K, V = qkv.split(C, dim=2)
        
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(out)
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=False),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
    
    def count_params(self):
        """Total trainable parameters, embeddings included."""
        return sum(p.numel() for p in self.parameters())

    def count_params_non_embedding(self):
        """Trainable parameters excluding token and position embeddings.

        This is the N used in scaling law fits. Kaplan et al. (2020, §2.1)
        exclude embeddings because the embedding table scales with vocab_size
        rather than with model capacity, so including it corrupts the N axis at
        small scale. Here the effect is extreme: at d_model=64 with a 16k vocab
        the token embedding alone is 84% of all parameters, versus 25% at
        d_model=512, so total-parameter counts would measure capacity confounded
        with a shrinking constant.

        The LM head is weight-tied to tok_emb and so is already excluded.
        """
        n = self.count_params()
        n -= self.tok_emb.weight.numel()
        n -= self.pos_emb.weight.numel()
        return n

