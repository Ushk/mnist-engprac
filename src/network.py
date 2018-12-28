from src.tensor import Tensor
from src.layers import Layer
from typing import Sequence
from typing import Dict

class NeuralNetwork:

    def __init__(self, layers: Sequence[Layer])-> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def zero_grad(self):
        for layer in reversed(self.layers):
            layer.grads: Dict[str, Tensor] = {}
