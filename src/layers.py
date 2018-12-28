import numpy as np
from typing import Dict
from src.tensor import Tensor

class Layer:

    def __init__(self) -> None:
        self.grads:  Dict[str, Tensor] = {}
        self.has_params = False

    def forward(self, inputs: Tensor)-> Tensor:
        raise NotImplementedError

    def backward(self, dl_dy: Tensor)-> Tensor:
        raise NotImplementedError

    def param_gradients(self, dl_dy: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super(Linear, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.has_params = True
        self.W : Tensor  = np.random.randn(self.num_inputs, self.num_outputs)*np.sqrt(2/self.num_outputs)
        self.b : Tensor = np.zeros(num_outputs)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computes a forward pass of the layer
        Assumes inputs has shape (batch_size, num_inputs)
        """
        assert inputs.shape[1] == self.num_inputs, f'inputs have incorrect shape. Expected (batch_size, {self.num_inputs}), got {inputs.shape}' 
        self.stored_activations = inputs
        return inputs @ self.W + self.b

    def backward(self, dl_dy: Tensor) -> Tensor:
        """
        Compute gradient of loss with respect to inputs
        y = w * x + b
        -> dy_dx = w
        :param dl_dy: gradient from upstream layer
        :return: dl_dx: gradient with respect to inputs
        """
        dl_dx = dl_dy @ self.W.T

        self.param_gradients(dl_dy)

        return dl_dx


    def param_gradients(self, dl_dy: Tensor) -> Tensor:
        self.grads['b'] = np.sum(dl_dy,axis=0)
        self.grads['W'] = self.stored_activations.T @ dl_dy
        del self.stored_activations


class Sigmoid(Layer):

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return 1.0 / (1.0 + np.exp(x))

    def backward(self, grad: Tensor) -> Tensor:
        return (self.forward(self.x) * self.forward(1 - self.x))*grad


class Relu(Layer):

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad: Tensor) -> Tensor:
        return (self.x > 0) * grad


class Softmax(Layer):

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Tensor with dimensions (Batch_size,num_classes)
        :return: Tensor with same dimensions as input, but softmaxed. Max subtracted for numerical stability
        """
        self.x = x
        e_x: Tensor = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, grad: Tensor) -> Tensor:

        assert (self.x.ndim == 2), f'x should be 2D, got {self.x.ndim} dimensions'

        ex = self.forward(self.x)

        batch_size, classes = np.shape(ex)

        identity = np.stack([np.eye(classes) for i in range(batch_size)])

        one_hot = np.expand_dims(grad* ex, axis=-1)

        offset = (identity - (np.expand_dims(ex, axis=-1)))

        new_grad : Tensor = np.matmul(offset, one_hot).reshape(batch_size, classes) # Removes extra dim

        return new_grad










