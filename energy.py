"""
Energy Module (V3.5 - Logical Body)

Goal: "Understanding" as Energy Minimization.
Method: JEPA (Joint Embedding Predictive Architecture).
Energy Function: E = Prediction Error + Logical Violation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPA_Predictor(nn.Module):
    """
    Predicts the next latent state given the current state and an action/context.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, D)
        action: (B, A) - Encoded action or context
        Returns: z_pred (B, D)
        """
        inp = torch.cat([z_t, action], dim=-1)
        return self.net(inp)

class EnergyFunction(nn.Module):
    """
    Computes the Energy (Loss) of the system.
    """
    def __init__(self, lambda_violation: float = 100.0):
        super().__init__()
        self.lambda_violation = lambda_violation

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor, violation_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes Energy.
        z_pred: (B, D) - Predicted next state
        z_target: (B, D) - Actual next state (Embedding of the observation)
        violation_mask: (B,) - 1.0 if Logical Violation (Contradiction/Error) occurred, 0.0 otherwise.
        """
        # 1. Prediction Error (Latent Distance)
        # We want the prediction to match the reality.
        pred_error = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=-1) # (B,)

        # 2. Logical Violation Penalty
        # If the action led to a contradiction (e.g. syntax error, assertion fail),
        # we impose a massive energy penalty.
        # This forces the agent to avoid "Thinking" in invalid ways.
        violation_energy = self.lambda_violation * violation_mask

        total_energy = pred_error + violation_energy
        
        return total_energy.mean()
