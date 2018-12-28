import numpy as np
from src.tensor import Tensor

class Loss:

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions-targets)**2

    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return 2*(predictions-targets)

class CrossEntropyLoss(Loss):

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        :param predictions: logits, dimensions (batch_size, nclasses)
        :param targets: actual labels, dimensions (batch_size, nclasses)
        :return: loss dimensions (batch_size), summed over classes.
        """
        return -1.0*np.sum(targets*np.log(predictions), axis=1)

    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return -targets*(1/predictions)
