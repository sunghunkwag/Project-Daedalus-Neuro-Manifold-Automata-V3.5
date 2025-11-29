"""
Project Daedalus V3.5: Logical Body (ASI Seed) - TRUE AUTONOMY

Main Orchestration Script.
Integrates:
- Soul (Axioms)
- Vision (Structure)
- Manifold (Reasoning)
- Energy (Goal)
- Automata (Memory)
- World (Sandbox)
- Action Decoder (Thought -> Code)

The Agent loops (REAL):
1. Perceive: Current state from Sandbox
2. Think: Brain processes state -> Action logits
3. Act: Sample action from logits, execute code
4. Observe: Parse result with Vision
5. Learn: Update network to minimize energy

"NO SCRIPT. ONLY ENERGY GUIDES THE WAY."
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List

from soul import get_soul_vectors
from vision import GNNObjectExtractor
from manifold import GraphAttentionManifold
from energy import EnergyFunction, JEPA_Predictor
from automata import ManifoldAutomata
from world import InternalSandbox
from action_decoder import ActionDecoder

class ASIAgent:
    def __init__(self):
        # 1. Soul Injection
        print("[System] Injecting Soul...")
        self.v_identity, self.v_truth, self.v_reject = get_soul_vectors(dim=32)
        
        # 2. Body Construction
        print("[System] Building Logical Body...")
        self.vision = GNNObjectExtractor(max_objects=5, feature_dim=4)
        self.brain = ManifoldAutomata(state_dim=32, num_heads=4, v_truth=self.v_truth)
        self.action_decoder = ActionDecoder(state_dim=32, num_actions=6)
        self.predictor = JEPA_Predictor(state_dim=32, action_dim=32)
        self.energy_fn = EnergyFunction(lambda_violation=100.0)
        self.world = InternalSandbox()
        
        # Optimizer (Brain + ActionDecoder + Predictor)
        self.optimizer = torch.optim.Adam(
            list(self.brain.parameters()) + 
            list(self.action_decoder.parameters()) + 
            list(self.predictor.parameters()), 
            lr=0.001
        )
        
        # Metrics
        self.action_history = []

    def run_cycle(self, step: int):
        print(f"\n{'='*60}")
        print(f"Cycle {step}")
        print(f"{'='*60}")
        
        # A. PERCEIVE: Get current state
        # Start with empty grid or use previous result
        if 'grid' in self.world.global_context:
            grid_np = self.world.global_context['grid']
            grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
            node_features, adjacency = self.vision(grid_tensor)
            print(f"[Perceive] Observed {node_features.shape[1]} objects from previous state")
        else:
            # Initial state: empty perception
            node_features = torch.zeros(1, 5, 4)
            adjacency = torch.eye(5).unsqueeze(0)
            print("[Perceive] Initial empty state")
        
        # B. THINK: Brain processes state -> Action logits
        input_state = torch.nn.functional.pad(node_features, (0, 28))  # 4 -> 32
        
        # Brain reasoning (multiple steps)
        current_state = self.brain(input_state, adjacency, steps=3)
        z_t = current_state.mean(dim=1)  # Global brain state (1, 32)
        
        # Get action logits from brain
        action_logits = self.action_decoder(z_t)  # (1, 6)
        
        # C. ACT: Sample action and execute
        # Exploration: stochastic sampling (early), exploitation: deterministic (late)
        deterministic = step > 15
        action_id = self.action_decoder.sample_action(action_logits.squeeze(0), deterministic=deterministic)
        action_name = self.action_decoder.get_action_name(action_id)
        
        print(f"[Think] Action Logits: {action_logits.squeeze(0).detach().numpy()}")
        print(f"[Act] Selected Action {action_id}: {action_name}")
        
        self.action_history.append(action_id)
        
        # Generate and execute code
        code = self.action_decoder.get_action_code(action_id)
        output, success = self.world.execute(code)
        
        if not success:
            print(f"[World] Execution FAILED: {output.strip()[:100]}")
        
        # D. OBSERVE: Parse result
        if success and 'grid' in self.world.global_context:
            grid_np = self.world.global_context['grid']
            grid_tensor = torch.tensor(grid_np, dtype=torch.float32)
            next_node_features, next_adjacency = self.vision(grid_tensor)
            print(f"[Observe] Extracted {next_node_features.shape[1]} objects")
        else:
            next_node_features = torch.zeros(1, 5, 4)
            next_adjacency = torch.eye(5).unsqueeze(0)
            print("[Observe] Nothing seen (execution failed)")
        
        # E. LEARN: Update network to minimize energy
        # Compute next state embedding
        next_input_state = torch.nn.functional.pad(next_node_features, (0, 28))
        next_state = self.brain(next_input_state, next_adjacency, steps=3)
        z_t1 = next_state.mean(dim=1)  # Next global state (1, 32)
        
        # Target: We want to be close to V_truth
        z_target = self.v_truth.unsqueeze(0)
        
        # Compute energy
        violation = 1.0 if not success else 0.0
        violation_tensor = torch.tensor([violation])
        
        # JEPA: Predict next state
        # Action embedding (one-hot)
        action_embedding = F.one_hot(torch.tensor([action_id]), num_classes=6).float()
        action_embedding = F.pad(action_embedding, (0, 26))  # 6 -> 32
        
        z_pred = self.predictor(z_t, action_embedding)
        
        # Energy = Prediction Error + Distance to Truth + Violation
        pred_error = F.mse_loss(z_pred, z_t1)
        truth_distance = F.mse_loss(z_t1, z_target)
        
        energy = pred_error + truth_distance + violation_tensor.squeeze() * 100.0
        
        # EWC Loss
        ewc = self.brain.ewc_loss()
        
        # Policy loss: Encourage actions that lead to low energy
        # We use the negative energy as "reward"
        # Higher log_prob for actions that resulted in low energy
        log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_prob = log_probs[0, action_id]
        
        # Policy gradient: maximize reward = minimize energy
        # Loss = -log_prob * (-energy) = log_prob * energy
        policy_loss = selected_log_prob * energy.detach()  # REINFORCE
        
        total_loss = energy + ewc - policy_loss  # Minimize energy, maximize policy reward
        
        print(f"[Heart] Energy: {energy.item():.4f} (Pred: {pred_error.item():.4f}, Truth: {truth_distance.item():.4f}, Violation: {violation})")
        print(f"[Memory] EWC Penalty: {ewc.item():.4f}")
        print(f"[Learn] Policy Loss: {policy_loss.item():.4f}")
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Crystallize if low energy and aligned with truth
        if success and truth_distance.item() < 0.05 and step == 18:
            print("[Memory] !!! CRYSTALLIZING KNOWLEDGE (Locking Weights) !!!")
            dummy_data = [(next_input_state, next_adjacency, z_target)]
            self.brain.register_ewc_task(dummy_data, lambda o, t: ((o.mean(1) - t)**2).mean())

def main():
    print("="*60)
    print("PROJECT DAEDALUS V3.5: AWAKENING (TRUE AUTONOMY)")
    print("="*60)
    print('"No script. Only energy guides the way."')
    print("="*60)
    
    agent = ASIAgent()
    
    for i in range(25):
        agent.run_cycle(i)
        
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Action History: {agent.action_history}")
    
    # Analyze action distribution
    import collections
    action_counts = collections.Counter(agent.action_history)
    print("\nAction Distribution:")
    for action_id in range(6):
        action_name = agent.action_decoder.get_action_name(action_id)
        count = action_counts.get(action_id, 0)
        print(f"  {action_id} ({action_name}): {count} times")

if __name__ == "__main__":
    main()
