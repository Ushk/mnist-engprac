from typing import List
from src.layers import Layer
from src.network import NeuralNetwork

class Optimizer:

    def step(self, net: NeuralNetwork)-> None:
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, net: NeuralNetwork, lr: float = 1e-3) -> None:
        super(SGD, self).__init__()
        self.lr = lr
        self.net_layers: List = [layer for layer in net.layers if layer.has_params is True]

    def step(self, net: NeuralNetwork) -> None:

        for layer in self.net_layers:
            layer.W -= self.lr*layer.grads['W']
            layer.b -= self.lr*layer.grads['b']




