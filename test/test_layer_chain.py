import numpy as np
from numpy.testing import assert_almost_equal
from itertools import product
import unittest

from eventprop.loss_layer import TTFSCrossEntropyLoss, TTFSCrossEntropyLossParameters
from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.eventprop_cpp import Spikes, SpikesVector
from test_lif_layer import get_normalization_factor, get_poisson_spikes


class LossLIFLIFChainTest(unittest.TestCase):
    def test_gradient_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 4
        n_batch = 3
        isi = 5e-3
        t_max = 0.2
        loss_pars = TTFSCrossEntropyLossParameters(n=4)
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(upper_pars.tau_mem, upper_pars.tau_syn)
        w_upper = np.random.normal(0.2, 0.1, size=(n_in, n_upper)) * norm_factor
        w_lower = np.random.normal(0.2, 0.1, size=(n_upper, n_lower)) * norm_factor
        w_eps = 1e-4
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        labels = np.arange(n_batch)
        grad_numerical_upper = np.zeros_like(w_upper)
        for batch_idx in range(n_batch):
            for idx in product(range(n_in), range(n_upper)):
                w_plus = np.copy(w_upper)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_plus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                loss_layer = TTFSCrossEntropyLoss(loss_pars)
                loss_layer(lower_layer(upper_layer(input_spikes)))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                loss_layer = TTFSCrossEntropyLoss(loss_pars)
                loss_layer(lower_layer(upper_layer(input_spikes)))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_upper[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        grad_numerical_lower = np.zeros_like(w_lower)
        for batch_idx in range(n_batch):
            for idx in product(range(n_upper), range(n_lower)):
                w_plus = np.copy(w_lower)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_plus)
                loss_layer = TTFSCrossEntropyLoss(loss_pars)
                loss_layer(lower_layer(upper_layer(input_spikes)))
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_minus)
                loss_layer = TTFSCrossEntropyLoss(loss_pars)
                loss_layer(lower_layer(upper_layer(input_spikes)))
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_lower[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        lower_layer = LIFLayer(lower_pars, w_lower)
        loss_layer = TTFSCrossEntropyLoss(loss_pars)
        loss_layer(lower_layer(upper_layer(input_spikes)))
        loss_layer.backward(labels)
        assert_almost_equal(grad_numerical_lower, lower_layer.gradient)
        assert_almost_equal(grad_numerical_upper, upper_layer.gradient)


class LIFLIFChainTest(unittest.TestCase):
    def test_gradient_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 3
        n_batch = 5
        isi = 10e-3
        t_max = 0.1
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(upper_pars.tau_mem, upper_pars.tau_syn)
        w_upper = np.random.normal(0.1, 0.01, size=(n_in, n_upper)) * norm_factor
        w_lower = np.random.normal(0.1, 0.01, size=(n_upper, n_lower)) * norm_factor
        w_eps = 1e-6
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        grad_numerical_upper = np.zeros_like(w_upper)
        for batch_idx in range(n_batch):
            for idx in product(range(n_in), range(n_upper)):
                w_plus = np.copy(w_upper)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_plus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_plus = np.sum(lower_layer.post_batch[batch_idx].times)

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_minus = np.sum(lower_layer.post_batch[batch_idx].times)

                grad_numerical_upper[idx] += (t_plus - t_minus) / (2 * w_eps)

        grad_numerical_lower = np.zeros_like(w_lower)
        for batch_idx in range(n_batch):
            for idx in product(range(n_upper), range(n_lower)):
                w_plus = np.copy(w_lower)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_plus)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_plus = np.sum(lower_layer.post_batch[batch_idx].times)

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_minus)
                lower_layer(upper_layer(input_spikes))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                t_minus = np.sum(lower_layer.post_batch[batch_idx].times)

                grad_numerical_lower[idx] += (t_plus - t_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        lower_layer = LIFLayer(lower_pars, w_lower)
        lower_layer(upper_layer(input_spikes))
        for spikes in lower_layer.post_batch:
            for spike_idx in range(spikes.n_spikes):
                spikes.set_error(spike_idx, 1)
        lower_layer.backward()
        assert_almost_equal(grad_numerical_lower, lower_layer.gradient)
        assert_almost_equal(grad_numerical_upper, upper_layer.gradient)


class LossLIFChainTest(unittest.TestCase):
    def test_gradient_vs_numerical_random(self):
        np.random.seed(0)
        n_in = 10
        n_neurons = 10
        n_batch = 5
        isi = 10e-3
        t_max = 0.1
        parameters = LIFLayerParameters(
            n=n_neurons, n_in=n_in, tau_mem=20e-3, tau_syn=10e-3
        )
        loss_params = TTFSCrossEntropyLossParameters(n=n_neurons, alpha=1)
        norm_factor = get_normalization_factor(parameters.tau_mem, parameters.tau_syn)
        w_in = np.random.normal(0.5, 0.1, size=(n_in, n_neurons)) * norm_factor
        w_eps = 1e-4
        input_spikes = SpikesVector(
            [get_poisson_spikes(isi, t_max, n_in) for _ in range(n_batch)]
        )
        labels = np.arange(n_batch)
        grad_numerical = np.zeros_like(w_in)
        for batch_idx in range(n_batch):
            for idx in product(range(n_in), range(n_neurons)):
                w_plus = np.copy(w_in)
                w_plus[idx] += w_eps
                layer = LIFLayer(parameters, w_plus)
                loss_layer = TTFSCrossEntropyLoss(loss_params)
                loss_layer(layer(input_spikes))
                assert layer.post_batch[batch_idx].n_spikes > 0
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_in)
                w_minus[idx] -= w_eps
                layer = LIFLayer(parameters, w_minus)
                loss_layer = TTFSCrossEntropyLoss(loss_params)
                loss_layer(layer(input_spikes))
                assert layer.post_batch[batch_idx].n_spikes > 0
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        layer = LIFLayer(parameters, w_in)
        loss_layer = TTFSCrossEntropyLoss(loss_params)
        loss_layer(layer(input_spikes))
        loss_layer.backward(labels)
        assert_almost_equal(grad_numerical, layer.gradient)

    def test_gradient_vs_numerical_single_post_spike(self):
        lif_params = LIFLayerParameters(n_in=1, n=11)
        loss_params = TTFSCrossEntropyLossParameters(n=11, alpha=1)
        norm_factor = get_normalization_factor(lif_params.tau_mem, lif_params.tau_syn)
        w_in = np.eye(lif_params.n_in, lif_params.n) * 1.1 * norm_factor
        input_spikes = SpikesVector(
            [Spikes(np.array([0], dtype=np.float64), np.array([0], dtype=np.int32))]
        )
        w_eps = 1e-8
        w_save = w_in[0, 0]

        w_in[0, 0] = w_save + w_eps
        loss_layer = TTFSCrossEntropyLoss(loss_params)
        lif_layer = LIFLayer(lif_params, w_in)
        loss_layer(lif_layer(input_spikes))
        assert lif_layer.post_batch[0].n_spikes > 0
        loss_plus = loss_layer.get_losses(0)[0]

        w_in[0, 0] = w_save - w_eps
        loss_layer = TTFSCrossEntropyLoss(loss_params)
        lif_layer = LIFLayer(lif_params, w_in)
        loss_layer(lif_layer(input_spikes))
        assert lif_layer.post_batch[0].n_spikes > 0
        loss_minus = loss_layer.get_losses(0)[0]

        numerical_grad = (loss_plus - loss_minus) / (2 * w_eps)

        loss_layer = TTFSCrossEntropyLoss(loss_params)
        lif_layer = LIFLayer(lif_params, w_in)
        loss_layer(lif_layer(input_spikes))
        loss_layer.backward([0])

        assert_almost_equal(numerical_grad, lif_layer.gradient[0, 0])

    def test_loss_lif_chain(self):
        lif_params = LIFLayerParameters(n_in=1, n=11)
        loss_params = TTFSCrossEntropyLossParameters(n=11)
        norm_factor = get_normalization_factor(lif_params.tau_mem, lif_params.tau_syn)
        w_in = np.eye(lif_params.n_in, lif_params.n) * 1.1 * norm_factor

        loss_layer = TTFSCrossEntropyLoss(loss_params)
        lif_layer = LIFLayer(lif_params, w_in)
        input_spikes = SpikesVector(
            [Spikes(np.array([0], dtype=np.float64), np.array([0], dtype=np.int32))]
        )

        loss_layer(lif_layer(input_spikes))
        loss_layer.backward([0])


if __name__ == "__main__":
    unittest.main()
