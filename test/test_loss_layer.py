import unittest
from itertools import product
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose

from eventprop.loss_layer import (
    TTFSCrossEntropyLoss,
    TTFSCrossEntropyLossParameters,
    VMaxCrossEntropyLoss,
)
from eventprop.eventprop_cpp import Spikes, SpikesVector
from eventprop.li_layer import LILayerParameters
from test_lif_layer import get_poisson_spikes


class TTFSCrossEntropyLossTest(unittest.TestCase):
    def test_numerical_gradient_vs_error(self):
        params = TTFSCrossEntropyLossParameters(n=5)

        def get_artificial_output(t_eps, batch_idx, spike_idx):
            times1 = np.arange(params.n) * 0.001
            times2 = times1 * 2
            if batch_idx == 0:
                times1[spike_idx] += t_eps
            elif batch_idx == 1:
                times2[spike_idx] += t_eps
            return SpikesVector(
                [
                    Spikes(
                        times1.astype(np.float64),
                        np.arange(params.n, dtype=np.int32),
                        times1.tolist(),
                        np.arange(params.n, dtype=np.int32).tolist(),
                    ),
                    Spikes(
                        times2.astype(np.float64),
                        np.arange(params.n, dtype=np.int32),
                        times2.tolist(),
                        np.arange(params.n, dtype=np.int32).tolist(),
                    ),
                ]
            )

        t_eps = 1e-8
        for label_neuron in range(params.n):
            batch_numerical_grads = list()
            for batch_idx in range(2):
                numerical_grads = list()
                for spike_idx in range(params.n):
                    loss = TTFSCrossEntropyLoss(params)
                    loss.forward(get_artificial_output(t_eps, batch_idx, spike_idx))
                    loss_plus = loss.get_losses([label_neuron])[batch_idx]

                    loss = TTFSCrossEntropyLoss(params)
                    loss.forward(get_artificial_output(-t_eps, batch_idx, spike_idx))
                    loss_minus = loss.get_losses([label_neuron])[batch_idx]

                    numerical_grads.append(
                        1 / 2 * (loss_plus - loss_minus) / (2 * t_eps)
                    )
                batch_numerical_grads.append(numerical_grads)

            loss = TTFSCrossEntropyLoss(params)
            loss.forward(get_artificial_output(0, 0, 0))
            loss.backward([label_neuron] * 2)
            for batch_idx in range(2):
                for spike_idx in range(params.n):
                    assert_allclose(
                        loss.input_batch[batch_idx].errors[spike_idx],
                        batch_numerical_grads[batch_idx][spike_idx],
                        rtol=1e-6,
                    )


class VMaxCrossEntropyLossTest(unittest.TestCase):
    def test_numerical_spike_gradient_vs_error(self):
        n_in = 10
        n_class = 10
        np.random.seed(0)
        w = np.random.normal(1, 1, size=(10, 10))
        params = LILayerParameters(n_in=n_in, n=n_class)
        # test spike gradient
        for label_neuron in range(params.n):
            artificial_output = SpikesVector(
                [
                    Spikes(
                        (np.arange(params.n) * 0.001).astype(np.float64),
                        np.arange(params.n, dtype=np.int32),
                    )
                ]
            )
            t_eps = 1e-8
            numerical_grads = list()
            for spike_idx in range(artificial_output[0].n_spikes):
                saved_time = artificial_output[0].times[spike_idx]
                artificial_output[0].set_time(spike_idx, saved_time + t_eps)
                loss = VMaxCrossEntropyLoss(params, w_in=w)
                loss.forward(artificial_output)
                loss_plus = loss.get_losses([label_neuron])

                artificial_output[0].set_time(spike_idx, saved_time - t_eps)
                loss = VMaxCrossEntropyLoss(params, w_in=w)
                loss.forward(artificial_output)
                loss_minus = loss.get_losses([label_neuron])
                artificial_output[0].set_time(spike_idx, saved_time)

                numerical_grads.append((loss_plus[0] - loss_minus[0]) / (2 * t_eps))

            loss = VMaxCrossEntropyLoss(params, w_in=w)
            loss.forward(artificial_output)
            loss.backward([label_neuron])
            for spike_idx in range(artificial_output[0].n_spikes):
                assert_almost_equal(
                    artificial_output[0].errors[spike_idx], numerical_grads[spike_idx]
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
        input_spikes = SpikesVector([get_poisson_spikes(isi, t_max, n_in)])
        for label in range(n_neurons):
            grad_numerical = np.zeros_like(w_in)
            for syn_idx, nrn_idx in product(range(n_in), range(n_neurons)):
                w_plus = np.copy(w_in)
                w_plus[syn_idx, nrn_idx] += w_eps
                layer = VMaxCrossEntropyLoss(parameters, w_plus)
                layer.forward(input_spikes)
                loss_plus = layer.get_losses([label])

                w_minus = np.copy(w_in)
                w_minus[syn_idx, nrn_idx] -= w_eps
                layer = VMaxCrossEntropyLoss(parameters, w_minus)
                layer.forward(input_spikes)
                loss_minus = layer.get_losses([label])

                grad_numerical[syn_idx, nrn_idx] = (loss_plus[0] - loss_minus[0]) / (
                    2 * w_eps
                )

            layer = VMaxCrossEntropyLoss(parameters, w_in)
            layer.forward(input_spikes)
            layer.backward([label])
            assert_almost_equal(grad_numerical, layer.gradient)


if __name__ == "__main__":
    unittest.main()
