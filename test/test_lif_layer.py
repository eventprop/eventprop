import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.layer import Spikes


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
    return Spikes(all_times, all_sources)


class LIFLayerCPPvsPythonTest(unittest.TestCase):
    def test_random_input(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        isi = 10e-3
        t_max = 0.1
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.random.normal(0.07, 0.01, size=(n_in, n_neurons)) * norm_factor
        input_spikes = get_poisson_spikes(isi, t_max, n_in)
        python_layer = LIFLayer(parameters, w_in)
        python_layer.forward(input_spikes, code="python")
        cpp_layer = LIFLayer(parameters, w_in)
        cpp_layer.forward(input_spikes, code="cpp")
        assert_almost_equal(cpp_layer.post_spikes.times, python_layer.post_spikes.times)
        assert_almost_equal(
            cpp_layer.post_spikes.errors, python_layer.post_spikes.errors
        )
        assert_almost_equal(
            cpp_layer.post_spikes.sources, python_layer.post_spikes.sources
        )

    def test_layer_wise_spike_finder_cpp_vs_python(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = Spikes(np.arange(n_in) * 0.1, np.arange(n_in))
        python_layer = LIFLayer(parameters, w_in)
        python_layer.forward(input_spikes, code="python")
        assert python_layer.post_spikes.n_spikes == 0
        cpp_layer = LIFLayer(parameters, w_in)
        cpp_layer.forward(input_spikes, code="cpp")
        assert cpp_layer.post_spikes.n_spikes == 0
        assert_almost_equal(cpp_layer.post_spikes.times, python_layer.post_spikes.times)
        assert_almost_equal(
            cpp_layer.post_spikes.errors, python_layer.post_spikes.errors
        )
        assert_almost_equal(
            cpp_layer.post_spikes.sources, python_layer.post_spikes.sources
        )

        w_in = np.eye(n_in, n_neurons) * 1.001 * norm_factor
        python_layer = LIFLayer(parameters, w_in)
        python_layer.forward(input_spikes, code="python")
        assert python_layer.post_spikes.n_spikes == n_in
        cpp_layer = LIFLayer(parameters, w_in)
        cpp_layer.forward(input_spikes, code="cpp")
        assert cpp_layer.post_spikes.n_spikes == n_in
        assert_almost_equal(cpp_layer.post_spikes.times, python_layer.post_spikes.times)
        assert_almost_equal(
            cpp_layer.post_spikes.errors, python_layer.post_spikes.errors
        )
        assert_almost_equal(
            cpp_layer.post_spikes.sources, python_layer.post_spikes.sources
        )


class LIFLayerTest(unittest.TestCase):
    def __init__(self, *args, code="python", **kwargs):
        super().__init__(*args, **kwargs)
        self.code = code

    def test_constructor(self):
        parameters = LIFLayerParameters(n=100, n_in=3)
        w_in = np.ones((3, 100))
        layer = LIFLayer(parameters, w_in)

    def test_backward(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = Spikes(np.full(n_in, 0.1), np.arange(n_in))
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.backward(code=self.code)
        assert layer.post_spikes.n_spikes == 0
        assert np.all(layer.gradient == 0)

        w_in = np.eye(n_in, n_neurons) * 1.00001 * norm_factor
        input_spikes = Spikes(np.full(n_in, 0.1), np.arange(n_in))
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, self.code)
        assert layer.post_spikes.n_spikes == n_in
        layer.backward(code=self.code)
        assert np.all(layer.gradient == 0)

        w_in = np.eye(n_in, n_neurons) * 1.0001 * norm_factor
        input_spikes = Spikes(np.full(n_in, 0.1), np.arange(n_in))
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == n_in

        a = w_in[0, 0]
        x = np.sqrt(a ** 2 - 4 * a * parameters.v_th)
        grad_analytical = (
            2 * parameters.tau_syn * (1 / a + 2 * parameters.v_th / ((a + x) * x))
            - 2 * parameters.tau_syn / x
        )
        layer.post_spikes.errors[:] = 1.00
        layer.backward(code=self.code)
        for grad in layer.gradient[:, 0]:
            assert_almost_equal(grad, grad_analytical)

    def test_backward_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        isi = 10e-3
        t_max = 0.1
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.random.normal(0.07, 0.01, size=(n_in, n_neurons)) * norm_factor
        w_eps = 1e-8
        input_spikes = get_poisson_spikes(isi, t_max, n_in)
        grad_numerical = np.zeros_like(w_in)
        for syn_idx in range(n_in):
            w_plus = np.copy(w_in)
            w_plus[syn_idx, :] += w_eps
            layer = LIFLayer(parameters, w_plus)
            layer.forward(input_spikes, code=self.code)
            t_plus = np.array(
                [
                    np.nansum(
                        layer.post_spikes.times[layer.post_spikes.sources == nrn_idx]
                    )
                    for nrn_idx in range(n_neurons)
                ]
            )

            w_minus = np.copy(w_in)
            w_minus[syn_idx, :] -= w_eps
            layer = LIFLayer(parameters, w_minus)
            layer.forward(input_spikes, code=self.code)
            t_minus = np.array(
                [
                    np.nansum(
                        layer.post_spikes.times[layer.post_spikes.sources == nrn_idx]
                    )
                    for nrn_idx in range(n_neurons)
                ]
            )

            grad_numerical[syn_idx, :] = (t_plus - t_minus) / (2 * w_eps)

        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.post_spikes.errors[:] = 1.00
        layer.backward(code=self.code)
        assert_almost_equal(grad_numerical, layer.gradient)

    def test_backward_vs_numerical_single_post_spike(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=7e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)

        w_eps = 1e-8
        # Test with single pre spike
        w_save = 1.2 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = Spikes(np.array([0.1]), np.array([0]))
        w_in[0, 0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 1
        t_plus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 1
        t_minus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 1
        layer.post_spikes.errors[0] = 1
        layer.backward(code=self.code)
        grad_numerical = (t_plus - t_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

        # Test with two pre spikes
        w_save = 0.6 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = Spikes(np.array([0.1, 0.105]), np.array([0, 0]))
        w_in[0, 0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 1
        t_plus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 1
        t_minus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 1
        layer.post_spikes.errors[0] = 1.00
        layer.backward(code=self.code)
        grad_numerical = (t_plus - t_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

    def test_backward_vs_numerical_two_post_spikes(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=7e-3
        )
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_eps = 1e-8

        w_save = 1.9 * norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0, 0] = w_save
        input_spikes = Spikes(np.array([0.1]), np.array([0]))
        # Test gradient for first post spike
        w_in[0, 0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        t_plus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        t_minus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        layer.post_spikes.errors[0] = 1
        layer.post_spikes.errors[1] = 0
        layer.backward(code=self.code)
        grad_numerical = (t_plus - t_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

        # Test gradient for second post spike
        w_in[0, 0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        t_plus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        t_minus = layer.post_spikes.times[0]
        w_in[0, 0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        layer.post_spikes.errors[1] = 0
        layer.post_spikes.errors[0] = 1
        layer.backward(code=self.code)
        grad_numerical = (t_plus - t_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

        # Test gradient for both post spikes
        w_in[0, 0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        t_plus = layer.post_spikes.times[0] + layer.post_spikes.times[1]
        w_in[0, 0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        t_minus = layer.post_spikes.times[0] + layer.post_spikes.times[1]
        w_in[0, 0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 2
        layer.post_spikes.errors[1] = 1
        layer.post_spikes.errors[0] = 1
        layer.backward(code=self.code)
        grad_numerical = (t_plus - t_minus) / (2 * w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0, 0])

    def test_spike_finder(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons) * norm_factor
        input_spikes = Spikes(np.arange(n_in) * 0.1, np.arange(n_in))
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == 0

        w_in = np.eye(n_in, n_neurons) * 1.001 * norm_factor
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert layer.post_spikes.n_spikes == n_in

    def test_current(self):
        parameters = LIFLayerParameters(n=1, n_in=1)
        w_in = np.ones((1, 1))
        input_spikes = Spikes(np.array([0.1]), np.array([0]))
        layer = LIFLayer(parameters, w_in)
        layer.forward(Spikes(np.array([]), np.array([])))
        i = layer._i(0.1 - 0.0001, 0)
        assert i == 0.0
        layer.forward(input_spikes, code=self.code)
        i = layer._i(0.1, 0)
        assert i == 1.0


class LIFLayerCPPTest(LIFLayerTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = "cpp"


if __name__ == "__main__":
    unittest.main()
