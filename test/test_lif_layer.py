import unittest
import numpy as np
import logging
import pickle
import os
from itertools import product
from numpy.testing import assert_almost_equal

from eventprop.lif_layer import LIFLayer, LIFLayerParameters, Spike

def get_normalization_factor(tau_mem : float, tau_syn : float) -> float:
    tau_factor = tau_mem / tau_syn
    return (tau_mem - tau_syn)/tau_syn * np.power(tau_factor, tau_factor/(tau_factor-1)) / (tau_factor-1)

def get_poisson_times(isi, t_max):
    times = [np.random.exponential(isi)]
    while times[-1] < t_max:
        times.append(times[-1]+np.random.exponential(isi))
    return times[:-1]

class LIFLayerCPPvsPythonTest(unittest.TestCase):
    def test_random_input(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        isi = 10e-3
        t_max = 0.1
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.random.normal(0.07, 0.01, size=(n_in, n_neurons))*norm_factor
        input_spikes = list()
        for nrn_idx in range(n_in):
            times = get_poisson_times(isi, t_max)
            input_spikes.extend([Spike(source_neuron=nrn_idx, time=t) for t in times])
        python_layer = LIFLayer(parameters, w_in)
        python_layer.forward(input_spikes, code="python")
        cpp_layer = LIFLayer(parameters, w_in)
        cpp_layer.forward(input_spikes, code="cpp")
        assert_almost_equal([x.time for x in cpp_layer.post_spikes], [x.time for x in python_layer.post_spikes])
        assert([x.error for x in cpp_layer.post_spikes] == [x.error for x in python_layer.post_spikes])
        assert([x.source_neuron for x in cpp_layer.post_spikes] == [x.source_neuron for x in python_layer.post_spikes])
        assert([len(x) for x in cpp_layer._post_spikes_per_neuron] == [len(x) for x in python_layer._post_spikes_per_neuron])

    def test_layer_wise_spike_finder_cpp_vs_python(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons)*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1*nrn_idx) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code="python")
        python_spikes = layer.post_spikes
        assert(len(layer.post_spikes) == 0)
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code="cpp")
        cpp_spikes = layer.post_spikes
        assert(len(layer.post_spikes) == 0)
        assert(python_spikes == cpp_spikes)

        w_in = np.eye(n_in, n_neurons)*1.001*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1*nrn_idx) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code="python")
        python_spikes = layer.post_spikes
        assert(len(layer.post_spikes) == n_in)
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code="cpp")
        cpp_spikes = layer.post_spikes
        assert(len(layer.post_spikes) == n_in)
        assert_almost_equal([x.time for x in python_spikes], [x.time for x in cpp_spikes])
        assert([x.source_neuron for x in python_spikes] == [x.source_neuron for x in cpp_spikes])

class LIFLayerTest(unittest.TestCase):
    def __init__(self, *args, code="python", **kwargs):
        super().__init__(*args, **kwargs)
        self.code = code

    def test_constructor(self):
        parameters = LIFLayerParameters(n=100, n_in=3)
        w_in = np.ones((3,100))
        layer = LIFLayer(parameters, w_in)

    def test_backward(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons)*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        layer.backward(layer.post_spikes, code=self.code)
        assert(len(layer.post_spikes) == 0)
        assert(np.all(layer.gradient == 0))

        w_in = np.eye(n_in, n_neurons)*1.00001*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, self.code)
        assert(len(layer.post_spikes) == n_in)
        layer.backward(layer.post_spikes, code=self.code)
        assert(np.all(layer.gradient == 0))

        w_in = np.eye(n_in, n_neurons)*1.0001*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == n_in)

        a = w_in[0,0]
        x = np.sqrt(a**2-4*a*parameters.v_th)
        grad_analytical = 2*parameters.tau_syn*(1/a + 2*parameters.v_th/((a+x)*x)) - 2*parameters.tau_syn/x
        for spike in layer.post_spikes:
            spike.error = 1
        layer.backward(layer.post_spikes, code=self.code)
        for grad in layer.gradient[:,0]:
            assert_almost_equal(grad, grad_analytical)

    def test_backward_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 5
        n_neurons = 10
        isi = 10e-3
        t_max = 0.1
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.random.normal(0.07, 0.01, size=(n_in, n_neurons))*norm_factor
        w_eps = 1e-8
        input_spikes = list()
        for nrn_idx in range(n_in):
            times = get_poisson_times(isi, t_max)
            input_spikes.extend([Spike(source_neuron=nrn_idx, time=t) for t in times])
        grad_numerical = np.zeros_like(w_in)
        for syn_idx in range(n_in):
            w_plus = np.copy(w_in)
            w_plus[syn_idx, :] += w_eps
            layer = LIFLayer(parameters, w_plus)
            layer.forward(input_spikes, code=self.code)
            t_plus = np.array([sum([spike.time for spike in layer._post_spikes_per_neuron[nrn_idx]]) for nrn_idx in range(n_neurons)])

            w_minus = np.copy(w_in)
            w_minus[syn_idx, :] -= w_eps
            layer = LIFLayer(parameters, w_minus)
            layer.forward(input_spikes, code=self.code)
            t_minus = np.array([sum([spike.time for spike in layer._post_spikes_per_neuron[nrn_idx]]) for nrn_idx in range(n_neurons)])

            grad_numerical[syn_idx, :] = (t_plus - t_minus)/(2*w_eps)

        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        for spike in layer.post_spikes:
            spike.error = 1.
        layer.backward(layer.post_spikes, code=self.code)
        assert_almost_equal(grad_numerical, layer.gradient)

    def test_backward_vs_numerical_single_post_spike(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=7e-3)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)

        w_eps = 1e-8
        # Test with single pre spike
        w_save = 1.2*norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0,0] = w_save
        input_spikes = [Spike(source_neuron=0, time=0.1)]
        w_in[0,0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 1)
        t_plus = layer.post_spikes[0].time
        w_in[0,0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 1)
        t_minus = layer.post_spikes[0].time
        w_in[0,0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 1)
        layer.post_spikes[0].error = 1
        layer.backward(layer.post_spikes, code=self.code)
        grad_numerical = (t_plus - t_minus)/(2*w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0,0])

        # Test with two pre spikes
        w_save = 0.6*norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0,0] = w_save
        input_spikes = [Spike(source_neuron=0, time=0.1), Spike(source_neuron=0, time=0.105)]
        w_in[0,0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 1)
        t_plus = layer.post_spikes[0].time
        w_in[0,0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 1)
        t_minus = layer.post_spikes[0].time
        w_in[0,0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 1)
        layer.post_spikes[0].error = 1
        layer.backward(layer.post_spikes, code=self.code)
        grad_numerical = (t_plus - t_minus)/(2*w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0,0])

    def test_backward_vs_numerical_two_post_spikes(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=7e-3)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_eps = 1e-8

        w_save = 1.9*norm_factor
        w_in = np.zeros((n_in, n_neurons))
        w_in[0,0] = w_save
        input_spikes = [Spike(source_neuron=0, time=0.1)]
        # Test gradient for first post spike
        w_in[0,0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        t_plus = layer.post_spikes[0].time
        w_in[0,0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        t_minus = layer.post_spikes[0].time
        w_in[0,0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        layer.post_spikes[0].error = 1
        layer.post_spikes[1].error = 0
        layer.backward(layer.post_spikes, code=self.code)
        grad_numerical = (t_plus - t_minus)/(2*w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0,0])

        # Test gradient for second post spike
        w_in[0,0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        t_plus = layer.post_spikes[0].time
        w_in[0,0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        t_minus = layer.post_spikes[0].time
        w_in[0,0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        layer.post_spikes[1].error = 0
        layer.post_spikes[0].error = 1
        layer.backward(layer.post_spikes, code=self.code)
        grad_numerical = (t_plus - t_minus)/(2*w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0,0])

        # Test gradient for both post spikes
        w_in[0,0] = w_save + w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        t_plus = layer.post_spikes[0].time + layer.post_spikes[1].time
        w_in[0,0] = w_save - w_eps
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        t_minus = layer.post_spikes[0].time + layer.post_spikes[1].time
        w_in[0,0] = w_save
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 2)
        layer.post_spikes[1].error = 1
        layer.post_spikes[0].error = 1
        layer.backward(layer.post_spikes, code=self.code)
        grad_numerical = (t_plus - t_minus)/(2*w_eps)
        assert_almost_equal(grad_numerical, layer.gradient[0,0])


    def test_spike_finder(self):
        n_in = 5
        n_neurons = 10
        parameters = LIFLayerParameters(n=n_neurons, n_in=n_in)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.eye(n_in, n_neurons)*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1*nrn_idx) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == 0)

        w_in = np.eye(n_in, n_neurons)*1.001*norm_factor
        input_spikes = [Spike(source_neuron=nrn_idx, time=0.1*nrn_idx) for nrn_idx in range(n_in)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(input_spikes, code=self.code)
        assert(len(layer.post_spikes) == n_in)

    def test_current(self):
        parameters = LIFLayerParameters(n=1, n_in=1)
        w_in = np.ones((1,1))
        input_spikes = [Spike(source_neuron=0, time=0.1)]
        layer = LIFLayer(parameters, w_in)
        layer.forward(list())
        i = layer._i(0.1-0.0001, 0)
        assert(i == 0.)
        layer.forward(input_spikes, code=self.code)
        i = layer._i(0.1, 0)
        assert(i == 1.)

class LIFLayerCPPTest(LIFLayerTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = "cpp"

    def test_pickle(self):
        spikes = list()
        for _ in range(100):
            time = np.random.random()
            source = np.random.randint(0, 10000000)
            error = np.random.random()
            spikes.append(Spike(time=time, source_neuron=source, error=error, source_layer=id(self)))
        with open("/tmp/spikes.pkl", "wb") as f:
            pickle.dump(spikes, f)
        with open("/tmp/spikes.pkl", "rb") as f:
            pickled_spikes = pickle.load(f)
        os.remove("/tmp/spikes.pkl")
        assert(pickled_spikes == spikes)


if __name__ == "__main__":
    unittest.main()