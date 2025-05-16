"""
Mixture of Experts (MoE) Implementation

This module implements a Mixture of Experts (MoE) model, which consists of:
1. Multiple expert networks that specialize on different parts of the input space
2. A gating network that determines which expert(s) to use for a given input
3. A mechanism to combine the outputs of the selected experts

The implementation includes both dense and sparse variants of MoE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    A single expert network implementation.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DenseMoE(nn.Module):
    """
    Dense Mixture of Experts where all experts process each input.
    This is computationally expensive but simpler to implement.
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size, dropout) 
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, ..., input_size]
            
        Returns:
            Tensor of shape [batch_size, ..., output_size]
        """
        # Store original shape for later reshaping
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # Reshape input if needed
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])
            
        # Compute gating weights
        gates = self.gate(x)  # Shape: [batch_size, num_experts]
        
        # Apply each expert to the input
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        # Shape: [num_experts, batch_size, output_size]
        
        # Combine expert outputs according to the gating weights
        # Reshape gates: [batch_size, num_experts, 1]
        gates_expanded = gates.unsqueeze(-1)
        
        # Transpose expert_outputs to [batch_size, num_experts, output_size]
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Weighted sum of expert outputs
        combined_output = torch.sum(expert_outputs * gates_expanded, dim=1)
        
        # Reshape back to original dimensions
        if len(original_shape) > 2:
            output_size = combined_output.shape[-1]
            combined_output = combined_output.reshape(*original_shape[:-1], output_size)
            
        return combined_output, gates


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts where only top-k experts process each input.
    This is more computationally efficient for large numbers of experts.
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts=8, 
                 k=2, capacity_factor=1.0, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)  # Can't select more experts than we have
        self.capacity_factor = capacity_factor
        
        # Create multiple expert networks
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size, dropout) 
            for _ in range(num_experts)
        ])
        
        # Gating network (no softmax, we'll apply it to top-k only)
        self.gate = nn.Linear(input_size, num_experts)
        
        # For tracking expert usage statistics
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        
    def _compute_capacity(self, batch_size):
        """
        Compute how many tokens each expert should process.
        """
        # Calculate the capacity of each expert
        tokens_per_expert = batch_size / self.num_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        capacity = max(capacity, 4)  # Minimum capacity
        return capacity
        
    def forward(self, x):
        """
        Forward pass through the Sparse MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, ..., input_size]
            
        Returns:
            Tensor of shape [batch_size, ..., output_size]
        """
        # Store original shape for later reshaping
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # Reshape input if needed to [batch_size, input_size]
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])
            batch_size = x.shape[0]
            
        # Get raw gating scores
        gate_logits = self.gate(x)  # Shape: [batch_size, num_experts]
        
        # Add noise to encourage exploration (optional)
        if self.training:
            noise = torch.randn_like(gate_logits) * 1e-2
            gate_logits += noise
            
        # Get top-k experts for each input
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1)
        
        # Normalize the probabilities of the top-k experts
        top_k_probs_normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate expert capacity
        capacity = self._compute_capacity(batch_size)
        
        # Create a mask for each expert and token
        # Shape: [batch_size, num_experts]
        expert_mask = torch.zeros(batch_size, self.num_experts, device=x.device)
        
        # Prepare to combine expert outputs
        combined_output = torch.zeros(batch_size, self.experts[0].fc2.out_features, device=x.device)
        
        # For tracking which experts were used
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which inputs use this expert
            token_indices = torch.nonzero(top_k_indices == expert_idx, as_tuple=True)[0]
            
            # Skip if no input uses this expert
            if len(token_indices) == 0:
                continue
                
            # Limit capacity if needed
            if len(token_indices) > capacity:
                token_indices = token_indices[:capacity]
                
            # Track expert usage
            expert_usage[expert_idx] = len(token_indices)
            
            # Get the corresponding inputs
            expert_inputs = x[token_indices]
            
            # Get the corresponding weights for this expert
            expert_probs = torch.zeros(len(token_indices), device=x.device)
            
            # Find where in the top-k this expert appears for each token
            for i, token_idx in enumerate(token_indices):
                # Find position of this expert in the top-k list for this token
                positions = (top_k_indices[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    pos = positions[0]
                    expert_probs[i] = top_k_probs_normalized[token_idx, pos]
            
            # Process inputs with this expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Combine the output weighted by the expert probability
            combined_output[token_indices] += expert_output * expert_probs.unsqueeze(1)
            
        # Update expert counts
        if self.training:
            self.expert_counts += expert_usage
            
        # Reshape back to original dimensions
        if len(original_shape) > 2:
            output_size = combined_output.shape[-1]
            combined_output = combined_output.reshape(*original_shape[:-1], output_size)
            
        return combined_output, gate_probs
        
    def load_balancing_loss(self, gates):
        """
        Compute the load balancing loss to encourage equal expert usage.
        
        Args:
            gates: Gate probabilities of shape [batch_size, num_experts]
            
        Returns:
            Load balancing loss (scalar tensor)
        """
        # Calculate the fraction of tokens routed to each expert
        routing_weights = gates.mean(dim=0)
        
        # Calculate the fraction of tokens routed to each expert
        # We want a uniform distribution
        target_routing = torch.ones_like(routing_weights) / self.num_experts
        
        # Compute the loss (mean squared error)
        loss = torch.sum((routing_weights - target_routing) ** 2)
        
        return loss


class MoELayer(nn.Module):
    """
    A general MoE layer that can be used as a drop-in replacement for a feed-forward layer.
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts=8, 
                 k=2, sparse=True, capacity_factor=1.0, dropout=0.1):
        super().__init__()
        
        # Choose between dense and sparse implementation
        if sparse:
            self.moe = SparseMoE(
                input_size, 
                hidden_size, 
                output_size, 
                num_experts=num_experts, 
                k=k,
                capacity_factor=capacity_factor,
                dropout=dropout
            )
        else:
            self.moe = DenseMoE(
                input_size, 
                hidden_size, 
                output_size, 
                num_experts=num_experts,
                dropout=dropout
            )
            
    def forward(self, x):
        output, gates = self.moe(x)
        return output, gates


class MoETransformerBlock(nn.Module):
    """
    A transformer block with a Mixture of Experts FFN replacing the standard FFN.
    """
    def __init__(self, d_model, nhead, d_ff, num_experts=8, k=2, dropout=0.1):
        super().__init__()
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # MoE layer instead of standard feed-forward
        self.moe = MoELayer(
            input_size=d_model,
            hidden_size=d_ff,
            output_size=d_model,
            num_experts=num_experts,
            k=k,
            sparse=True,
            dropout=dropout
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention block (with residual connection)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # MoE block (with residual connection)
        moe_output, gates = self.moe(x)
        x = x + self.dropout(moe_output)
        x = self.norm2(x)
        
        return x, gates


class SimpleMoEModel(nn.Module):
    """
    A simple model using MoE for demonstration purposes.
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts=8, k=2):
        super().__init__()
        
        self.embedding = nn.Linear(input_size, hidden_size)
        self.moe_layer = MoELayer(hidden_size, hidden_size*4, hidden_size, 
                                  num_experts=num_experts, k=k)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x, gates = self.moe_layer(x)
        x = self.output_layer(x)
        return x, gates 