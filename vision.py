"""
Vision Module (V3.5 - Logical Body)

Goal: Extract structure from raw data without CNNs.
Method: Algorithmic parsing (DFS/BFS) to identify objects and their relationships.
Output: Graph representation (Nodes, Adjacency).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict

class ObjectNode:
    def __init__(self, obj_id: int, category: int, centroid: Tuple[float, float], size: int):
        self.obj_id = obj_id
        self.category = category
        self.centroid = centroid
        self.size = size
        # Feature vector: [x, y, size, category] (normalized later)
        self.features = torch.tensor([centroid[0], centroid[1], size, category], dtype=torch.float32)

class GNNObjectExtractor(nn.Module):
    def __init__(self, max_objects: int = 10, feature_dim: int = 4):
        super().__init__()
        self.max_objects = max_objects
        self.feature_dim = feature_dim

    def forward(self, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parses a 2D grid into a graph.
        Input: grid (H, W) - Integer tensor where values represent categories/colors. 0 is background.
        Output:
            - node_features: (B, N, D)
            - adjacency: (B, N, N)
        """
        # Handle batch dimension
        if grid.dim() == 2:
            grid = grid.unsqueeze(0)
        
        B, H, W = grid.shape
        device = grid.device
        
        batch_nodes = []
        batch_adj = []

        for b in range(B):
            img = grid[b].cpu().numpy()
            nodes = self._extract_objects_dfs(img)
            
            # Pad or truncate to max_objects
            num_nodes = len(nodes)
            if num_nodes > self.max_objects:
                nodes = nodes[:self.max_objects]
            
            # Create feature tensor
            features = torch.zeros(self.max_objects, self.feature_dim, device=device)
            for i, node in enumerate(nodes):
                # Normalize features roughly
                norm_feat = node.features.clone()
                norm_feat[0] /= W # x
                norm_feat[1] /= H # y
                norm_feat[2] /= (W*H) # size
                # category is kept as is or embedded later. Here we just pass it.
                features[i] = norm_feat.to(device)
                
            # Create Adjacency Matrix (Spatial Distance based)
            adj = torch.eye(self.max_objects, device=device) # Self-loops
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # Euclidean distance between centroids
                    dist = np.sqrt((nodes[i].centroid[0] - nodes[j].centroid[0])**2 + 
                                   (nodes[i].centroid[1] - nodes[j].centroid[1])**2)
                    
                    # Simple logic: Closer objects are related
                    # Normalize distance by diagonal
                    max_dist = np.sqrt(H**2 + W**2)
                    norm_dist = dist / max_dist
                    
                    # Edge weight = 1 - distance (Closer = Stronger)
                    weight = max(0, 1.0 - norm_dist)
                    
                    adj[i, j] = weight
                    adj[j, i] = weight
            
            batch_nodes.append(features)
            batch_adj.append(adj)
            
        return torch.stack(batch_nodes), torch.stack(batch_adj)

    def _extract_objects_dfs(self, grid: np.ndarray) -> List[ObjectNode]:
        """
        Standard Connected Components Labeling using DFS/BFS.
        """
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        objects = []
        obj_id_counter = 0
        
        for y in range(H):
            for x in range(W):
                val = grid[y, x]
                if val != 0 and not visited[y, x]:
                    # Found new object
                    component_pixels = []
                    stack = [(x, y)]
                    visited[y, x] = True
                    
                    while stack:
                        cx, cy = stack.pop()
                        component_pixels.append((cx, cy))
                        
                        # Check 4-neighbors
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < W and 0 <= ny < H:
                                if not visited[ny, nx] and grid[ny, nx] == val:
                                    visited[ny, nx] = True
                                    stack.append((nx, ny))
                    
                    # Create ObjectNode
                    size = len(component_pixels)
                    # Centroid
                    sum_x = sum(p[0] for p in component_pixels)
                    sum_y = sum(p[1] for p in component_pixels)
                    centroid = (sum_x / size, sum_y / size)
                    
                    obj = ObjectNode(obj_id_counter, val, centroid, size)
                    objects.append(obj)
                    obj_id_counter += 1
                    
        return objects
