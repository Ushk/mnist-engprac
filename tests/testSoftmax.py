from unittest import TestCase
import numpy as np
from src.layers import Softmax

class TestLinear(TestCase):

    def setUp(self):
        self.nclasses = 5
        self.batch_size = 5
        self.softmax = Softmax()

    def test_forward(self):

        test_data = np.random.rand(self.batch_size, self.nclasses)

        test_data = self.softmax.forward(test_data)

        assert np.all(np.round(test_data.sum(axis=1),5)==1.)

        assert np.all( (test_data>0) & (test_data<1) )

    def test_backward(self):
        pass



