from unittest import TestCase
import numpy as np
from src.layers import Linear

class TestLinear(TestCase):

    def setUp(self):
        self.ninput = 32
        self.noutput = 16
        self.batch_size = 5
        self.linear = Linear(self.ninput, self.noutput)

    def test_forward(self):

        test_data = np.random.rand(self.batch_size,self.ninput)

        output = self.linear.forward(test_data)

        np.testing.assert_allclose(self.linear.stored_activations ,test_data)

        assert output.shape == (self.batch_size, self.noutput)

    def test_backward(self):

        test_data = np.random.rand(self.batch_size,self.ninput)

        self.linear.forward(test_data)

        test_grad = np.random.rand(self.batch_size, self.noutput)

        test_grad = self.linear.backward(test_grad)

        # Check grads is populated
        assert len(self.linear.grads) == 2

        # Check
        assert test_grad.shape == (self.batch_size, self.ninput)



