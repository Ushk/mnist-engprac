import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader as DataLoader
from typing import List, Tuple

from src.tensor import Tensor
from src.network import NeuralNetwork
from src.loss import Loss
from src.optim import Optimizer, SGD
from src.utils import RunningMean, classification_accuracy

NCLASSES=10
NINPUTS=784

def convert_to_onehot(t_indices: Tensor, nclasses: int):
    batch_size = t_indices.shape[0]
    oh_indices = np.zeros((batch_size, nclasses))
    oh_indices[np.arange(batch_size), t_indices] = 1
    return oh_indices


def run_experiment(net: NeuralNetwork,
                   data: DataLoader,
                   criterion: Loss,
                   optimizer: Optimizer,
                   train: bool = True,
                   ) -> Tuple[RunningMean, RunningMean]:

    loss = RunningMean()
    acc = RunningMean()

    for idx, (inputs, targets) in tqdm(enumerate(data)):

        inputs = inputs.reshape(-1, NINPUTS)
        inputs = inputs.numpy()

        targets = targets.numpy()
        targets = convert_to_onehot(targets, nclasses=NCLASSES)

        logits = net.forward(inputs)
        loss.update(criterion.forward(logits, targets).mean())
        acc.update(classification_accuracy(logits, targets))

        if train is True:
            net.zero_grad()
            loss_prime = criterion.backward(logits, targets)
            net.backward(loss_prime)
            optimizer.step(net)


    print(loss.quantity, acc.quantity)

    return loss, acc



