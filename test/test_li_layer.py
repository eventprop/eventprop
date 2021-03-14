import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing._private.utils import assert_equal

from eventprop.li_layer import LILayer, LILayerParameters
from eventprop.eventprop_cpp import Spikes, SpikesVector, Maxima, MaximaVector
from test_lif_layer import get_poisson_spikes, get_normalization_factor


class LILayerTest(unittest.TestCase):
    def test_constructor(self):
        parameters = LILayerParameters(n=100, n_in=3)
        w_in = np.ones((3, 100))
        layer = LILayer(parameters, w_in)

    def test_vmax(self):
        n_in = 5
        n_neurons = 10
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        t_max = (
            (parameters.tau_mem * parameters.tau_syn)
            / (parameters.tau_mem - parameters.tau_syn)
            * np.log(parameters.tau_mem / parameters.tau_syn)
        )
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = SpikesVector(
            [
                Spikes(
                    np.full(n_in, 0.1, dtype=np.float64),
                    np.arange(n_in, dtype=np.int32),
                )
            ]
        )
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        for max_idx in range(n_in):
            assert_almost_equal(layer.maxima_batch[0].values[max_idx], 1.0)
            assert_almost_equal(layer.maxima_batch[0].times[max_idx], 0.1 + t_max)
            assert_equal(layer.maxima_batch[0].errors[max_idx], 0)
        for max_idx in range(n_in, n_neurons):
            assert_almost_equal(layer.maxima_batch[0].values[max_idx], 0)
            assert np.isnan(layer.maxima_batch[0].times[max_idx])
            assert_equal(layer.maxima_batch[0].errors[max_idx], 0)

        input_spikes = SpikesVector(
            [
                Spikes(
                    np.concatenate(
                        [
                            np.full(n_in, 0.05, dtype=np.float64),
                            np.full(n_in, 0.1, dtype=np.float64),
                        ]
                    ),
                    np.concatenate(
                        [
                            np.arange(n_in, dtype=np.int32),
                            np.arange(n_in, dtype=np.int32),
                        ]
                    ),
                )
            ]
        )
        t_max = (
            (parameters.tau_mem * parameters.tau_syn)
            / (parameters.tau_mem - parameters.tau_syn)
            * np.log(
                parameters.tau_mem
                / parameters.tau_syn
                * (
                    (
                        np.exp(0.05 / parameters.tau_syn)
                        + np.exp(0.1 / parameters.tau_syn)
                    )
                    / (
                        np.exp(0.05 / parameters.tau_mem)
                        + np.exp(0.1 / parameters.tau_mem)
                    )
                )
            )
        )
        k = (
            lambda t: parameters.tau_syn
            / (parameters.tau_mem - parameters.tau_syn)
            * norm_factor
            * (np.exp(-t / parameters.tau_mem) - np.exp(-t / parameters.tau_syn))
        )
        target_v_max = k(t_max - 0.05) + k(t_max - 0.1)
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        for max_idx in range(n_in):
            assert_almost_equal(layer.maxima_batch[0].values[max_idx], target_v_max)
            assert_almost_equal(layer.maxima_batch[0].times[max_idx], t_max)
            assert_equal(layer.maxima_batch[0].errors[max_idx], 0)
        for max_idx in range(n_in, n_neurons):
            assert_equal(layer.maxima_batch[0].values[max_idx], 0)
            assert np.isnan(layer.maxima_batch[0].times[max_idx])
            assert_equal(layer.maxima_batch[0].errors[max_idx], 0)

    def test_backward(self):
        n_in = 5
        n_neurons = 10
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = SpikesVector(
            [
                Spikes(
                    np.full(n_in, 0.1, dtype=np.float64),
                    np.arange(n_in, dtype=np.int32),
                )
            ]
        )
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        layer.backward()
        assert np.all(layer.gradient == 0)
        assert np.all(layer.input_batch[0].errors == 0)

        w_in = np.eye(n_in, n_neurons) * 1.0001 * norm_factor
        input_spikes = SpikesVector(
            [
                Spikes(
                    np.full(n_in, 0.1, dtype=np.float64),
                    np.arange(n_in, dtype=np.int32),
                )
            ]
        )
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)

        grad_analytical = 1 / norm_factor
        for max_idx in range(n_neurons):
            layer.maxima_batch[0].set_error(max_idx, 1.0)
        layer.backward()
        assert_almost_equal(
            layer.input_batch[0].errors, np.zeros_like(layer.input_batch[0].errors)
        )
        for grad in layer.gradient[:, 0]:
            assert_almost_equal(grad, grad_analytical)

    def test_backward_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        n_batch = 5
        isi = 10e-3
        t_max = 0.1
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        w_in = np.random.normal(1, 1, size=(n_in, n_neurons))
        w_eps = 1e-7
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        grad_numerical = np.zeros_like(w_in)
        for syn_idx in range(n_in):
            w_plus = np.copy(w_in)
            w_plus[syn_idx, :] += w_eps
            layer = LILayer(parameters, w_plus)
            layer.forward(input_spikes)
            v_plus = np.sum([x.values for x in layer.maxima_batch], axis=0)

            w_minus = np.copy(w_in)
            w_minus[syn_idx, :] -= w_eps
            layer = LILayer(parameters, w_minus)
            layer.forward(input_spikes)
            v_minus = np.sum([x.values for x in layer.maxima_batch], axis=0)

            grad_numerical[syn_idx, :] = (v_plus - v_minus) / (2 * w_eps)

        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        for batch_idx in range(n_batch):
            for max_idx in range(n_neurons):
                layer.maxima_batch[batch_idx].set_error(max_idx, 1.0)
        layer.backward()
        assert_almost_equal(grad_numerical, layer.gradient)

    def test_backward_vs_numerical_single_post_spike(self):
        n_in = 5
        n_neurons = 10
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=7e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)

        w_eps = 1e-8
        # Test with single pre spike
        w_save = 1.2 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = SpikesVector(
            [Spikes(np.array([0.1], dtype=np.float64), np.array([0], dtype=np.int32))]
        )
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_plus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_minus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        layer.maxima_batch[0].set_error(0, 1)
        layer.backward()
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

        # Test with two pre spikes
        w_save = 0.6 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = SpikesVector(
            [
                Spikes(
                    np.array([0.1, 0.105], dtype=np.float64),
                    np.array([0, 0], dtype=np.int32),
                )
            ]
        )
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_plus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_minus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        layer.maxima_batch[0].set_error(0, 1.0)
        layer.backward()
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

    def test_backward_vs_numerical_two_post_spikes(self):
        n_in = 5
        n_neurons = 10
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=7e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_eps = 1e-8

        w_save = 1.9 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = SpikesVector(
            [Spikes(np.array([0.1], dtype=np.float64), np.array([0], dtype=np.int32))]
        )
        # Test gradient for first post spike
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_plus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_minus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        layer.maxima_batch[0].set_error(0, 1)
        layer.maxima_batch[0].set_error(1, 0)
        layer.backward()
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])
        np.all(layer.input_batch[0].errors == 0)

        w_in[0, 1] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_plus = layer.maxima_batch[0].values[1]
        w_in[0, 1] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_minus = layer.maxima_batch[0].values[1]
        w_in[0, 1] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        layer.maxima_batch[0].set_error(0, 0)
        layer.maxima_batch[0].set_error(1, 1)
        layer.backward()
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 1])
        np.all(layer.input_batch[0].errors == 0)

        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_plus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        v_minus = layer.maxima_batch[0].values[0]
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes)
        layer.maxima_batch[0].set_error(0, 1)
        layer.maxima_batch[0].set_error(1, 1)
        layer.backward()
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])
        np.all(layer.input_batch[0].errors == 0)


if __name__ == "__main__":
    unittest.main()
