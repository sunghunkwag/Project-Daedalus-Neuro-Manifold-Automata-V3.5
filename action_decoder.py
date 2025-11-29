"""
Action Decoder Module (V3.5 - True Autonomy)

Translates abstract brain outputs (vectors) into concrete Python code.
This is the bridge between "Thought" and "Action".
"""

import torch
import torch.nn as nn

class ActionDecoder(nn.Module):
    """
    Converts brain state into discrete action primitives.
    The agent learns which actions minimize energy.
    """
    def __init__(self, state_dim: int, num_actions: int = 6):
        super().__init__()
        self.num_actions = num_actions
        
        # Brain output -> Action logits
        self.policy_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, num_actions)
        )
        
        # Action primitives (Code templates)
        self.action_primitives = {
            0: self._action_do_nothing,
            1: self._action_random_noise,
            2: self._action_draw_square,
            3: self._action_symmetrize,
            4: self._action_clear,
            5: self._action_draw_pair
        }
        
    def forward(self, brain_state: torch.Tensor) -> torch.Tensor:
        """
        brain_state: (B, D) - Global state from the brain
        Returns: action_logits (B, num_actions)
        """
        logits = self.policy_head(brain_state)
        return logits
    
    def sample_action(self, logits: torch.Tensor, deterministic: bool = False) -> int:
        """
        Sample an action ID from the logits.
        """
        if deterministic:
            action_id = torch.argmax(logits, dim=-1).item()
        else:
            # Stochastic sampling (exploration)
            probs = torch.softmax(logits, dim=-1)
            action_id = torch.multinomial(probs, num_samples=1).item()
        
        return action_id
    
    def get_action_code(self, action_id: int, grid_shape: tuple = (10, 10)) -> str:
        """
        Convert action ID to executable Python code.
        """
        if action_id in self.action_primitives:
            return self.action_primitives[action_id](grid_shape)
        else:
            return self._action_do_nothing(grid_shape)
    
    # ===== Action Primitives =====
    
    def _action_do_nothing(self, grid_shape):
        return """
import numpy as np
grid = np.zeros({}, dtype=int)
print("Action: Do Nothing")
""".format(grid_shape)
    
    def _action_random_noise(self, grid_shape):
        return """
import numpy as np
grid = np.random.randint(0, 2, {})
print("Action: Random Noise")
""".format(grid_shape)
    
    def _action_draw_square(self, grid_shape):
        h, w = grid_shape
        x, y = h // 4, w // 4
        size = min(h, w) // 3
        return """
import numpy as np
grid = np.zeros({}, dtype=int)
grid[{}:{}, {}:{}] = 1
print("Action: Draw Square")
""".format(grid_shape, x, x+size, y, y+size)
    
    def _action_symmetrize(self, grid_shape):
        return """
import numpy as np
grid = np.random.randint(0, 2, {})
# Create symmetry (Truth-aligned behavior)
grid = (grid + grid.T) // 2
print("Action: Symmetrize")
""".format(grid_shape)
    
    def _action_clear(self, grid_shape):
        return """
import numpy as np
grid = np.zeros({}, dtype=int)
print("Action: Clear")
""".format(grid_shape)
    
    def _action_draw_pair(self, grid_shape):
        h, w = grid_shape
        x, y = h // 4, w // 4
        size = min(h, w) // 4
        return """
import numpy as np
grid = np.zeros({}, dtype=int)
grid[{}:{}, {}:{}] = 1  # Left Square
grid[{}:{}, {}:{}] = 1  # Right Square (Symmetry)
print("Action: Draw Symmetric Pair")
""".format(grid_shape, x, x+size, y, y+size, x, x+size, w-y-size, w-y)

    def get_action_name(self, action_id: int) -> str:
        """
        Get human-readable action name.
        """
        names = {
            0: "Do Nothing",
            1: "Random Noise",
            2: "Draw Square",
            3: "Symmetrize",
            4: "Clear",
            5: "Draw Pair"
        }
        return names.get(action_id, "Unknown")
