"""
Automata Module (V3.5 - Logical Body)

Goal: Lock 3 (Plasticity Control) via Elastic Weight Consolidation (EWC).
Method:
- Use GraphAttentionManifold for reasoning.
- Implement EWC to "Crystallize" important logic (e.g. 1+1=2).
- Prevent catastrophic forgetting of Axioms.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from manifold import GraphAttentionManifold

class ManifoldAutomata(nn.Module):
    def __init__(self, state_dim: int, num_heads: int = 4, v_truth: torch.Tensor = None):
        super().__init__()
        self.manifold = GraphAttentionManifold(state_dim, num_heads, v_truth)
        
        # Output/Update Layer (The "Thought" Process)
        self.update_net = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.LayerNorm(state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim)
        )
        
        # EWC Storage
        self.fisher_matrix = {} # Importance of each parameter
        self.opt_params = {}    # Optimal parameter values for previous tasks
        self.ewc_lambda = 1000.0 # Strength of crystallization

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Iterative reasoning on the manifold.
        """
        curr_x = x
        for _ in range(steps):
            # 1. Manifold Attention (Reasoning)
            # Nodes exchange information based on logical relevance
            context = self.manifold(curr_x, adjacency)
            
            # 2. State Update (Thinking)
            delta = self.update_net(context)
            curr_x = curr_x + delta
            
        return curr_x

    def register_ewc_task(self, dataset, loss_fn):
        """
        Compute Fisher Information Matrix for the current task and freeze weights.
        This is called after learning a "Truth" (e.g. 1+1=2).
        """
        self.eval()
        fisher = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p.data)

        # Estimate Fisher Information (Diagonal)
        # We need a dataset of "True" examples
        for data in dataset:
            self.zero_grad()
            # Forward pass
            # Assuming data is (x, adj, target)
            x, adj, target = data
            output = self(x, adj)
            loss = loss_fn(output, target)
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(dataset)
        
        # Store params
        for n, p in self.named_parameters():
            if p.requires_grad:
                self.fisher_matrix[n] = fisher[n]
                self.opt_params[n] = p.data.clone()
                
        self.train()

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute the EWC penalty loss.
        L_ewc = sum(F_i * (theta_i - theta_i_star)^2)
        """
        loss = torch.tensor(0.0)
        for n, p in self.named_parameters():
            if n in self.fisher_matrix:
                fisher = self.fisher_matrix[n]
                opt_param = self.opt_params[n]
                loss = loss + (fisher * (p - opt_param) ** 2).sum()
        
        return self.ewc_lambda * loss
