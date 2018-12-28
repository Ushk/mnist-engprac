import numpy as np
from src.tensor import Tensor

class Activation:
    
    def forward(self, x: Tensor) -> Tensor:
        pass
    
    def backward(self, x: Tensor) -> Tensor:
        pass


        