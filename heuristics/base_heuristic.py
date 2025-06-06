import numpy as np
from abc import ABC, abstractmethod

class Heuristic(ABC):
    def __init__(self, hierarchy):
        """
        Common initializer: Initialize the metric with a hierarchy object.
        """    
        self.hierarchy = hierarchy


    @abstractmethod
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Given node‐wise predictions of shape `(n_samples, n_nodes)`, produce
        a boolean/binary (0/1) vector for each sample of shape `(n_samples, n_nodes)`.
        """
        pass
    
    