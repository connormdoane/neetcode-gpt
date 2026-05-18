import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        epsilon = 1e-5
        mean = np.mean(x)
        var = np.var(x)
        x_norm = (x - mean) / np.sqrt(var + epsilon)
        return np.round(gamma * x_norm + beta, 5)
