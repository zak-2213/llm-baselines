"""
Llama style Language Model.
References:
1) Llama inference code:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
2) Mistral one file ref:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
3) Llama paper:
https://arxiv.org/pdf/2302.13971.pdf
 
Main differences from GPT2:
* Uses RMSNorm instead of LayerNorm
* Uses a slightly different MLP (SwiGLU)
* rotary embeddings (RoPE)
"""

import math

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.base import CausalSelfAttention, GPTBase

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, T, nh, hs)
    # freq_cis: (T, hs)
    # return: (B, T, nh, hs), (B, T, nh, hs)
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, q_)
    xq_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return xq_out.type_as(q), xk_out.type_as(k)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.n_embd * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(nn.functional.silu(self.w1(x)) * self.w2(x))


class LlamaAttention(CausalSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.rpe = torchtune.modules.RotaryPositionalEmbeddings(config.n_embd//config.n_head, config.sequence_length)
        if getattr(config, 'use_gqa', False):
            self.num_key_value_heads = config.num_key_value_heads
            # IMPORTANT: head_dim is based on n_head (Q's heads) for RoPE compatibility
            self.gqa_head_dim = config.n_embd // config.n_head

            self.q_proj_gqa = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.k_proj_gqa = nn.Linear(config.n_embd, self.num_key_value_heads * self.gqa_head_dim, bias=config.bias)
            self.v_proj_gqa = nn.Linear(config.n_embd, self.num_key_value_heads * self.gqa_head_dim, bias=config.bias)
    def forward(self, x, freqs_cis):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_head_dim = C // self.n_head
        if getattr(self.config, 'use_gqa', False):
            num_kv_heads = self.num_key_value_heads  # from LlamaAttention.__init__

            q = self.q_proj_gqa(x)
            k = self.k_proj_gqa(x)
            v = self.v_proj_gqa(x)

            # q shape: (B, T, n_head, query_head_dim)
            q = q.view(B, T, self.n_head, query_head_dim)
            # k, v shape: (B, T, num_kv_heads, query_head_dim) - use query_head_dim
            k = k.view(B, T, num_kv_heads, query_head_dim)
            v = v.view(B, T, num_kv_heads, query_head_dim)

            q, k = apply_rotary_emb(q, k, freqs_cis)
            # q = self.rpe(q)
            # k = self.rpe(k)
            # (B, nh, T, hs)
            q, k = q.transpose(1, 2), k.transpose(1, 2)

            # (B, nh, T, hs)
            v = v.transpose(1, 2)
            if num_kv_heads != self.n_head:
                repeat_factor = self.n_head // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            else:  # Standard MHA path
                # calculate query, key, values for all heads in batch and move head forward to be the batch dim
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # c_attn from CausalSelfAttention
    
                # (B, T, nh, hs) where hs = query_head_dim
                k = k.view(B, T, self.n_head, query_head_dim)
                q = q.view(B, T, self.n_head, query_head_dim)
                v = v.view(B, T, self.n_head, query_head_dim)  # v also uses query_head_dim
    
                q, k = apply_rotary_emb(q, k, freqs_cis)
                # (B, nh, T, hs)
                q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = LlamaAttention(config)
        self.ln_2 = torch.nn.RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


class Llama(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.sequence_length
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        # shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        for block_idx, block in enumerate(self.transformer.h):
            x = block(x, freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }
