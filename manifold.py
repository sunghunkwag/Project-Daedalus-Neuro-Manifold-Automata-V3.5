"""
Manifold Module (V3.5 - Logical Body)

Goal: Replace continuous Riemannian space with relational Logical space.
Method: Graph Attention Network (GAT) where 'distance' is 'logical relevance'.
Soul Injection: Truth Vector acts as an Attention Bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GraphAttentionManifold(nn.Module):
    """
    The "Space" of the Logical Body.
    It is not a fixed coordinate system, but a dynamic graph of relationships.
    """
    def __init__(self, state_dim: int, num_heads: int = 4, v_truth: Optional[torch.Tensor] = None):
        super().__init__()
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads
        
        assert self.head_dim * num_heads == state_dim, "state_dim must be divisible by num_heads"

        # GAT Layers
        self.W_query = nn.Linear(state_dim, state_dim)
        self.W_key = nn.Linear(state_dim, state_dim)
        self.W_value = nn.Linear(state_dim, state_dim)
        
        # Soul Bias Projector
        # Projects the relation (h_i, h_j) into the Truth Space
        if v_truth is not None:
            self.register_buffer('v_truth', v_truth)
            self.has_soul = True
        else:
            self.register_buffer('v_truth', torch.zeros(state_dim))
            self.has_soul = False
            
        self.soul_gate = nn.Linear(state_dim * 2, 1) # Gates the soul influence

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D) - Node features
        adjacency: (B, N, N) - Structural adjacency (from Vision)
        """
        B, N, D = x.shape
        
        # 1. Linear Projections
        Q = self.W_query(x).view(B, N, self.num_heads, self.head_dim)
        K = self.W_key(x).view(B, N, self.num_heads, self.head_dim)
        V = self.W_value(x).view(B, N, self.num_heads, self.head_dim)
        
        # Transpose for attention: (B, Heads, N, HeadDim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # Scores: (B, Heads, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 3. Soul Injection (Attention Bias)
        if self.has_soul:
            # We want to bias connections that are "True".
            # What is a "True" connection? One that aligns with the Truth Vector.
            # We compute a "Logical Consistency" score for each pair.
            
            # Expand x for pairwise comparison
            x_i = x.unsqueeze(2).expand(B, N, N, D)
            x_j = x.unsqueeze(1).expand(B, N, N, D)
            
            # Concatenate pair features: (B, N, N, 2D)
            pair_features = torch.cat([x_i, x_j], dim=-1)
            
            # Compute alignment with Truth (simplified)
            # We assume Truth is a direction in the parameter space.
            # Here we just use a learned gate that is initialized/regularized by Truth?
            # Or better: Project the pair difference onto V_truth?
            
            # "Symmetry" implies x_i should be related to x_j in a specific way.
            # Let's use the Soul Gate to compute a scalar bias.
            soul_bias = self.soul_gate(pair_features).squeeze(-1) # (B, N, N)
            
            # Add to scores (broadcast over heads)
            scores = scores + soul_bias.unsqueeze(1)

        # 4. Masking with Structural Adjacency
        # We only allow attention where vision detected a potential relation (or we can allow full attention)
        # For "Causal Flow", we might mask future nodes if it was a sequence, but here it's a set.
        # We use the adjacency as a "Soft Prior".
        # adjacency is (B, N, N).
        scores = scores + torch.log(adjacency.unsqueeze(1) + 1e-9)
        
        # 5. Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 6. Aggregation
        out = torch.matmul(attn_weights, V) # (B, Heads, N, HeadDim)
        
        # 7. Concatenate Heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        # Residual Connection + Norm (Standard Transformer block part, but simplified here)
        return out + x

