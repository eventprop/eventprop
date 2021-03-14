import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.eventprop_cpp import Spikes, SpikesVector


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


def get_poisson_spikes(isi, t_max, n):
    all_times = np.array([])
    all_sources = np.array([])
    for nrn_idx in range(n):
        times = get_poisson_times(isi, t_max)
        all_times = np.concatenate([all_times, times])
        all_sources = np.concatenate([all_sources, np.full(len(times), nrn_idx)])
    sort_idxs = np.argsort(all_times)
    all_times = all_times[sort_idxs]
    all_sources = all_sources[sort_idxs]
    return Spikes(
        all_times.astype(np.float64),
        all_sources.astype(np.int32),
    )


class LIFLayerCPPTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_spike_finder(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = SpikesVector(
            [
                Spikes(
                    (np.arange(n_in) * 0.1).astype(np.float64),
                    np.arange(n_in).astype(np.int32),
                ),
                Spikes(
                    (np.arange(n_in) * 0.1).astype(np.float64),
                    np.arange(n_in).astype(np.int32),
                ),
            ]
        )
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes)
        assert all([x.n_spikes == 0 for x in layer.post_batch])

        w_in = np.eye(n_in, n_neurons) * 1.001 * norm_factor
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes)
        assert all([x.n_spikes == n_in for x in layer.post_batch])

    def test_backward_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        n_batch = 10
        isi = 10e-3
        t_max = 0.1
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.random.normal(0.07, 0.01, size=(n_in, n_neurons)) * norm_factor
        w_eps = 1e-8
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        grad_numerical = np.zeros_like(w_in)
        for syn_idx in range(n_in):
            w_plus = np.copy(w_in)
            w_plus[syn_idx, :] += w_eps
            layer = LIFLayer(parameters, w_plus)
            layer.forward(input_spikes)
            t_plus = np.array(
                [
                    np.nansum(
                        [
                            np.nansum(x.times[x.sources == nrn_idx])
                            for x in layer.post_batch
                        ]
                    )
                    for nrn_idx in range(n_neurons)
                ]
            )

            w_minus = np.copy(w_in)
            w_minus[syn_idx, :] -= w_eps
            layer = LIFLayer(parameters, w_minus)
            layer.forward(input_spikes)
            t_minus = np.array(
                [
                    np.nansum(
                        [
                            np.nansum(x.times[x.sources == nrn_idx])
                            for x in layer.post_batch
                        ]
                    )
                    for nrn_idx in range(n_neurons)
                ]
            )

            grad_numerical[syn_idx, :] = (t_plus - t_minus) / (2 * w_eps)

        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes)
        for spikes in layer.post_batch:
            for spike_idx in range(spikes.n_spikes):
                spikes.set_error(spike_idx, 1)
        layer.backward()
        assert_almost_equal(grad_numerical, layer.gradient)


if __name__ == "__main__":
    unittest.main()
