from typing import List

from src.data import training_data, test_data
from src.loss import CrossEntropyLoss
from src.optim import SGD
from src.layers import Layer, Linear, Relu, Softmax
from src.network import NeuralNetwork
from src.run import run_experiment

layers: List[Layer] = [Linear(784,32), Relu(), Linear(32,32), Relu(), Linear(32,10), Softmax()]
net = NeuralNetwork(layers)
sgd = SGD(net)
run_experiment(net, training_data, CrossEntropyLoss(), sgd)










