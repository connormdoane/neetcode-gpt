import numpy as np
from numpy.typing import NDArray


class Solution:
    
    # Basic operations on numpy arrays are applied element-wise by default
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, z)
