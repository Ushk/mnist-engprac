from src.tensor import Tensor

class RunningMean:

    def __init__(self) -> None:
        self.quantity = 0
        self.idx = 0

    def update(self, batch_quantity:Tensor) -> None:
        self.idx += 1
        self.quantity = (self.quantity*(self.idx-1) + batch_quantity)/self.idx



def classification_accuracy(logits: Tensor, targets: Tensor) -> Tensor:
    pred_label =  logits.argmax(axis=1)
    return (pred_label == targets.argmax(axis=1)).mean()
