"""
Transformer-based model with Mixture of Experts for language modeling tasks.

This implementation shows how to replace standard feed-forward layers in transformer
architecture with MoE layers for improved scaling and parameter efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import tiktoken

from models.moe import MoELayer


class MoETransformerConfig:
    """Configuration class for MoETransformer model."""
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 vocabulary size
        sequence_length=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True,
        activation='gelu',
        # MoE specific parameters
        num_experts=8,
        top_k=2,
        capacity_factor=1.0,
        # For which layers to use MoE
        moe_layers=None,  # None means all, otherwise list of layer indices
    ):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        
        # MoE specific parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.moe_layers = moe_layers if moe_layers is not None else list(range(n_layer))


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create constant positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MoETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with Mixture of Experts feed-forward network.
    """
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config.n_embd
        self.nhead = config.n_head
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            config.n_embd, 
            config.n_head, 
            dropout=config.dropout
        )
        
        # MoE feed-forward layer (replaces standard feed-forward)
        self.moe_layer = MoELayer(
            input_size=config.n_embd,
            hidden_size=4 * config.n_embd,  # Standard transformer FF size
            output_size=config.n_embd,
            num_experts=config.num_experts,
            k=config.top_k,
            sparse=True,
            capacity_factor=config.capacity_factor,
            dropout=config.dropout
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # MoE feed-forward block
        src2 = self.norm2(src)
        src2, gates = self.moe_layer(src2)
        src = src + self.dropout2(src2)
        
        return src, gates


class StandardTransformerEncoderLayer(nn.Module):
    """
    Standard transformer encoder layer with regular feed-forward network.
    """
    def __init__(self, config):
        super().__init__()
        
        self.d_model = config.n_embd
        self.nhead = config.n_head
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            config.n_embd, 
            config.n_head, 
            dropout=config.dropout
        )
        
        # Standard feed-forward network
        self.linear1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.linear2 = nn.Linear(4 * config.n_embd, config.n_embd)
        
        # Activation function
        if config.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        # Return None for gates to match MoE layer interface
        return src, None


class MoETransformer(nn.Module):
    """
    Transformer language model with some layers replaced by MoE layers.
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2") if config.vocab_size == 50257 else None
        
        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_encoder = PositionalEncoding(config.n_embd, config.sequence_length)
        
        # Transformer layers (mix of standard and MoE)
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(config) if i in config.moe_layers
            else StandardTransformerEncoderLayer(config)
            for i in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(config.n_embd)
        
        # Output projection
        self.output_projection = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie embedding weights
        self.embedding.weight = self.output_projection.weight
        
        # Initialize parameters
        self._init_parameters()
        
        # Count parameters
        print(f"Total parameters: {self.get_num_params():,}")
        print(f"MoE parameters: {self.get_moe_params():,}")
        print(f"Non-MoE parameters: {self.get_num_params() - self.get_moe_params():,}")
        
    def get_num_params(self):
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_moe_params(self):
        """Return the number of parameters in MoE layers."""
        return sum(p.numel() for name, p in self.named_parameters() 
                  if 'moe_layer' in name)
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, targets=None):
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input tensor of token ids [batch_size, seq_len]
            targets: Optional target tensor for computing loss [batch_size, seq_len]
            
        Returns:
            dict with 'logits' and 'loss' keys
        """
        # Get sequence dimensions
        device = x.device
        b, t = x.size()
        assert t <= self.config.sequence_length, f"Input sequence too long: {t} > {self.config.sequence_length}"
        
        # Token and position embeddings
        token_embeddings = self.embedding(x).transpose(0, 1)  # [seq_len, batch_size, embedding_dim]
        x = self.pos_encoder(token_embeddings)
        
        # Create attention mask (causal)
        mask = torch.ones(t, t, device=device).triu_(1) * float('-inf')
        
        # Track routing probabilities for debugging
        routing_logits = []
        
        # Process all transformer layers
        for layer in self.layers:
            x, gates = layer(x, src_mask=mask)
            if gates is not None:
                routing_logits.append(gates)
        
        # Final norm and projection to vocabulary
        x = self.norm(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, embedding_dim]
        
        # Get logits
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        # Calculate loss if targets are provided
        if targets is not None:
            # Reshape logits and targets for loss calculation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            loss = None
            
        return {
            'logits': logits,
            'loss': loss,
            'routing_logits': routing_logits if routing_logits else None
        }
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text given a starting sequence of tokens.
        
        Args:
            idx: Initial token sequence [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If specified, restricts sampling to the top k most likely tokens
            
        Returns:
            Generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop the sequence to the maximum length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            
            # Get predictions
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            # Focus on the last token
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


def moe_load_balancing_loss(routing_logits, num_experts):
    """
    Compute the load balancing loss for MoE layers.
    
    Args:
        routing_logits: List of gate values from MoE layers
        num_experts: Number of experts in each MoE layer
        
    Returns:
        Load balancing loss
    """
    if not routing_logits:
        return torch.tensor(0.0)
    
    loss = torch.tensor(0.0, device=routing_logits[0].device)
    
    for gates in routing_logits:
        # Gates shape: [batch_size, seq_len, num_experts]
        # Calculate the fraction of routing to each expert
        routing_weights = gates.mean(dim=[0, 1])
        
        # Ideal balanced routing
        target_routing = torch.ones_like(routing_weights) / num_experts
        
        # Mean squared error
        loss += torch.sum((routing_weights - target_routing) ** 2)
    
    return loss / len(routing_logits) 