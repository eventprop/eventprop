from test.test_lif_layer import get_poisson_spikes
import unittest
from itertools import product
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from eventprop.loss_layer import (
    TTFSCrossEntropyLoss,
    TTFSCrossEntropyLossParameters,
    VMaxCrossEntropyLoss,
)
from eventprop.layer import Spikes
from eventprop.li_layer import LILayerParameters
from test_lif_layer import get_poisson_spikes


class TTFSCrossEntropyLossTest(unittest.TestCase):
    def test_numerical_gradient_vs_error(self):
        params = TTFSCrossEntropyLossParameters()
        artificial_output = Spikes(np.arange(params.n) * 0.001, np.arange(params.n))
        for label_neuron in range(params.n):
            t_eps = 1e-9
            numerical_grads = list()
            for spike_idx in range(params.n):
                saved_time = artificial_output.times[spike_idx]
                artificial_output.times[spike_idx] += t_eps
                loss = TTFSCrossEntropyLoss(params)
                loss.forward(artificial_output)
                loss_plus = loss.get_loss(label_neuron)
                artificial_output.times[spike_idx] = saved_time

                artificial_output.times[spike_idx] -= t_eps
                loss = TTFSCrossEntropyLoss(params)
                loss.forward(artificial_output)
                loss_minus = loss.get_loss(label_neuron)
                artificial_output.times[spike_idx] = saved_time

                numerical_grads.append((loss_plus - loss_minus) / (2 * t_eps))

            loss = TTFSCrossEntropyLoss(params)
            loss.forward(artificial_output)
            loss.backward(label_neuron)
            for spike_idx in range(params.n):
                assert_allclose(
                    artificial_output.errors[spike_idx],
                    numerical_grads[spike_idx],
                    rtol=1e-6,
                )


class VMaxCrossEntropyLossTest(unittest.TestCase):
    def __init__(self, *args, code: str = "python", **kwargs):
        super().__init__(*args, **kwargs)
        self.code = code

    def test_numerical_vmax_gradient_vs_error(self):
        n_in = 10
        n_class = 10
        np.random.seed(0)
        w = np.random.normal(1, 1, size=(10, 10))
        params = LILayerParameters(n_in=n_in, n=n_class)
        artificial_output = Spikes(np.arange(params.n) * 0.001, np.arange(params.n))
        # test vmax gradient
        for label_neuron in range(params.n):
            v_eps = 1e-6
            numerical_grads = list()
            loss = VMaxCrossEntropyLoss(params, w_in=w)
            loss.forward(artificial_output)
            for vmax in loss.vmax:
                saved_vmax = vmax.value
                vmax.value += v_eps
                loss_plus = loss.get_loss(label_neuron)
                vmax.value = saved_vmax

                vmax.value -= v_eps
                loss_minus = loss.get_loss(label_neuron)
                numerical_grads.append((loss_plus - loss_minus) / (2 * v_eps))

            loss = VMaxCrossEntropyLoss(params, w_in=w)
            loss.forward(artificial_output)
            loss.backward(label_neuron)
            for vmax, num_grad in zip(loss.vmax, numerical_grads):
                assert_almost_equal(vmax.error, num_grad)

    def test_numerical_spike_gradient_vs_error(self):
        n_in = 10
        n_class = 10
        np.random.seed(0)
        w = np.random.normal(1, 1, size=(10, 10))
        params = LILayerParameters(n_in=n_in, n=n_class)
        # test spike gradient
        for label_neuron in range(params.n):
            artificial_output = Spikes(np.arange(params.n) * 0.001, np.arange(params.n))
            t_eps = 1e-8
            numerical_grads = list()
            for spike_idx in range(artificial_output.n_spikes):
                saved_time = artificial_output.times[spike_idx]
                artificial_output.times[spike_idx] += t_eps
                loss = VMaxCrossEntropyLoss(params, w_in=w)
                loss.forward(artificial_output)
                loss_plus = loss.get_loss(label_neuron)
                artificial_output.times[spike_idx] = saved_time

                artificial_output.times[spike_idx] -= t_eps
                loss = VMaxCrossEntropyLoss(params, w_in=w)
                loss.forward(artificial_output)
                loss_minus = loss.get_loss(label_neuron)
                artificial_output.times[spike_idx] = saved_time

                numerical_grads.append((loss_plus - loss_minus) / (2 * t_eps))

            loss = VMaxCrossEntropyLoss(params, w_in=w)
            loss.forward(artificial_output)
            loss.backward(label_neuron)
            for spike_idx in range(artificial_output.n_spikes):
                assert_almost_equal(
                    artificial_output.errors[spike_idx], numerical_grads[spike_idx]
                )

    def test_numerical_weight_gradient_vs_error(self):
        np.random.seed(0)
        n_in = 3
        n_neurons = 5
        isi = 10e-3
        t_max = 0.1
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        w_in = np.random.normal(1, 1, size=(n_in, n_neurons))
        w_eps = 1e-8
        input_spikes = get_poisson_spikes(isi, t_max, n_in)
        for label in range(n_neurons):
            grad_numerical = np.zeros_like(w_in)
            for syn_idx, nrn_idx in product(range(n_in), range(n_neurons)):
                w_plus = np.copy(w_in)
                w_plus[syn_idx, nrn_idx] += w_eps
                layer = VMaxCrossEntropyLoss(parameters, w_plus)
                layer.forward(input_spikes, code=self.code)
                loss_plus = layer.get_loss(label)

                w_minus = np.copy(w_in)
                w_minus[syn_idx, nrn_idx] -= w_eps
                layer = VMaxCrossEntropyLoss(parameters, w_minus)
                layer.forward(input_spikes, code=self.code)
                loss_minus = layer.get_loss(label)

                grad_numerical[syn_idx, nrn_idx] = (loss_plus - loss_minus) / (
                    2 * w_eps
                )

            layer = VMaxCrossEntropyLoss(parameters, w_in)
            layer.forward(input_spikes, code=self.code)
            layer.backward(label, code=self.code)
            assert_almost_equal(grad_numerical, layer.gradient)


if __name__ == "__main__":
    unittest.main()
