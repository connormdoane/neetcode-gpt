import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / (fan_in + fan_out))
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0
            weights.append(torch.randn(dims[i + 1], dims[i]) * std)
        
        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = torch.relu(x @ w.T)
            stds.append(round(x.std().item(), 2))
        
        return stds
            
