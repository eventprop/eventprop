import unittest
import numpy as np
import logging
import pickle
import os
from itertools import product
from numpy.testing import assert_almost_equal
from numpy.testing._private.utils import assert_equal

from eventprop.li_layer import LILayer, LILayerParameters, Spike


def get_normalization_factor(tau_mem: float, tau_syn: float) -> float:
    tau_factor = tau_mem / tau_syn
    return (
        (tau_mem - tau_syn)
        / tau_syn
        * np.power(tau_factor, tau_factor / (tau_factor - 1))
        / (tau_factor - 1)
    )


def get_poisson_times(isi, t_max):
    times = [np.random.exponential(isi)]
    while times[-1] < t_max:
        times.append(times[-1] + np.random.exponential(isi))
    return times[:-1]


class LILayerTest(unittest.TestCase):
    def __init__(self, *args, code="python", **kwargs):
        super().__init__(*args, **kwargs)
        self.code = code

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
        input_spikes = [
            Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)
        ]
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        for vmax in layer.vmax[:n_in]:
            assert_almost_equal(vmax.value, 1.0)
            assert_almost_equal(vmax.time, 0.1 + t_max)
            assert_equal(vmax.error, 0)
        for vmax in layer.vmax[n_in:]:
            assert_almost_equal(vmax.value, 0)
            assert vmax.time is None
            assert_equal(vmax.error, 0)

        input_spikes = [
            Spike(source_neuron=nrn_idx, time=0.05) for nrn_idx in range(n_in)
        ]
        input_spikes += [
            Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)
        ]
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
        layer.forward(input_spikes, code=self.code)
        for vmax in layer.vmax[:n_in]:
            assert_almost_equal(vmax.value, target_v_max)
            assert_almost_equal(vmax.time, t_max)
            assert_equal(vmax.error, 0)
        for vmax in layer.vmax[n_in:]:
            assert vmax.time is None
            assert_equal(vmax.error, 0)

    def test_backward(self):
        n_in = 5
        n_neurons = 10
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = [
            Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)
        ]
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.backward(code=self.code)
        assert np.all(layer.gradient == 0)
        assert np.all([spike.error == 0 for spike in input_spikes])

        w_in = np.eye(n_in, n_neurons) * 1.0001 * norm_factor
        input_spikes = [
            Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)
        ]
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)

        grad_analytical = 1 / norm_factor
        for vmax in layer.vmax:
            vmax.error = 1
        layer.backward(code=self.code)
        assert_almost_equal(
            [spike.error for spike in input_spikes], np.zeros(len(input_spikes))
        )
        for grad in layer.gradient[:, 0]:
            assert_almost_equal(grad, grad_analytical)

    def test_backward_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        isi = 10e-3
        t_max = 0.1
        parameters = LILayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        w_in = np.random.normal(1, 1, size=(n_in, n_neurons))
        w_eps = 1e-8
        input_spikes = list()
        for nrn_idx in range(n_in):
            times = get_poisson_times(isi, t_max)
            input_spikes.extend([Spike(source_neuron=nrn_idx, time=t) for t in times])
        grad_numerical = np.zeros_like(w_in)
        for syn_idx in range(n_in):
            w_plus = np.copy(w_in)
            w_plus[syn_idx, :] += w_eps
            layer = LILayer(parameters, w_plus)
            layer.forward(input_spikes, code=self.code)
            v_plus = np.array([vmax.value for vmax in layer.vmax])

            w_minus = np.copy(w_in)
            w_minus[syn_idx, :] -= w_eps
            layer = LILayer(parameters, w_minus)
            layer.forward(input_spikes, code=self.code)
            v_minus = np.array([vmax.value for vmax in layer.vmax])

            grad_numerical[syn_idx, :] = (v_plus - v_minus) / (2 * w_eps)

        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        for vmax in layer.vmax:
            vmax.error = 1.0
        layer.backward(code=self.code)
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
        input_spikes = [Spike(source_neuron=0, time=0.1)]
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_plus = layer.vmax[0].value
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_minus = layer.vmax[0].value
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.vmax[0].error = 1
        layer.backward(code=self.code)
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])
        assert_almost_equal(
            [spike.error for spike in input_spikes], np.zeros(len(input_spikes))
        )

        # Test with two pre spikes
        w_save = 0.6 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = [
            Spike(source_neuron=0, time=0.1),
            Spike(source_neuron=0, time=0.105),
        ]
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_plus = layer.vmax[0].value
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_minus = layer.vmax[0].value
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.vmax[0].error = 1
        layer.backward(code=self.code)
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
        input_spikes = [Spike(source_neuron=0, time=0.1)]
        # Test gradient for first post spike
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_plus = layer.vmax[0].value
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_minus = layer.vmax[0].value
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.vmax[0].error = 1
        layer.vmax[1].error = 0
        layer.backward(code=self.code)
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])
        assert_almost_equal(
            [spike.error for spike in input_spikes], np.zeros(len(input_spikes))
        )

        # Test gradient for second post spike
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_plus = layer.vmax[0].value
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_minus = layer.vmax[0].value
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.vmax[1].error = 0
        layer.vmax[0].error = 1
        layer.backward(code=self.code)
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])
        assert_almost_equal(
            [spike.error for spike in input_spikes], np.zeros(len(input_spikes))
        )

        # Test gradient for both post spikes
        w_in[0, 0] = w_save + w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_plus = layer.vmax[0].value + layer.vmax[1].value
        w_in[0, 0] = w_save - w_eps
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        v_minus = layer.vmax[0].value + layer.vmax[1].value
        w_in[0, 0] = w_save
        layer = LILayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.vmax[1].error = 1
        layer.vmax[0].error = 1
        layer.backward(code=self.code)
        grad_numerical = (v_plus - v_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])
        assert_almost_equal(
            [spike.error for spike in input_spikes], np.zeros(len(input_spikes))
        )


if __name__ == "__main__":
    unittest.main()
