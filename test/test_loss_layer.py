import unittest
import numpy as np
from copy import deepcopy
from numpy.testing import assert_almost_equal, assert_allclose

from eventprop.loss_layer import TTFSCrossEntropyLoss, TTFSCrossEntropyLossParameters
from eventprop.layer import Spike


class LossLayerTest(unittest.TestCase):
    def test_numerical_gradient_vs_error(self):
        params = TTFSCrossEntropyLossParameters()
        artificial_output = list()
        for idx in range(params.n):
            artificial_output.append(Spike(source_neuron=idx, time=0.001 * (idx)))
        for label_neuron in range(params.n):
            t_eps = 1e-9
            numerical_grads = list()
            for spike in artificial_output:
                saved_time = spike.time
                spike.time += t_eps
                loss = TTFSCrossEntropyLoss(params)
                loss.forward(artificial_output)
                loss_plus = loss.get_loss(label_neuron)
                spike.time = saved_time

                spike.time -= t_eps
                loss = TTFSCrossEntropyLoss(params)
                loss.forward(artificial_output)
                loss_minus = loss.get_loss(label_neuron)
                spike.time = saved_time

                numerical_grads.append((loss_plus - loss_minus) / (2 * t_eps))

            loss = TTFSCrossEntropyLoss(params)
            loss.forward(artificial_output)
            loss.backward(label_neuron)
            for spike, num_grad in zip(artificial_output, numerical_grads):
                assert_allclose(spike.error, num_grad, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
