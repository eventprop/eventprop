import numpy as np
from numpy.testing import assert_almost_equal
from itertools import product
import unittest

from eventprop.loss_layer import (
    TTFSCrossEntropyLoss,
    TTFSCrossEntropyLossParameters,
    VMaxCrossEntropyLoss,
    VMaxCrossEntropyLossParameters,
)
from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.eventprop_cpp import Spikes, SpikesVector
from test_lif_layer import get_normalization_factor, get_poisson_spikes


class LossLIFLIFChainTest(unittest.TestCase):
    def test_gradient_vs_numerical_random_ttfs(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 4
        n_batch = 3
        isi = 5e-3
        t_max = 0.2
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        loss_pars = TTFSCrossEntropyLossParameters(lif_parameters=lower_pars)
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
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_lower)
                loss_layer(upper_layer(input_spikes))
                assert loss_layer.post_batch[batch_idx].n_spikes > 0
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_lower)
                loss_layer(upper_layer(input_spikes))
                assert loss_layer.post_batch[batch_idx].n_spikes > 0
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_upper[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        grad_numerical_lower = np.zeros_like(w_lower)
        for batch_idx in range(n_batch):
            for idx in product(range(n_upper), range(n_lower)):
                w_plus = np.copy(w_lower)
                w_plus[idx] += w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_plus)
                loss_layer(upper_layer(input_spikes))
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_minus)
                loss_layer(upper_layer(input_spikes))
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_lower[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        loss_layer = TTFSCrossEntropyLoss(loss_pars, w_in=w_lower)
        loss_layer(upper_layer(input_spikes))
        loss_layer.backward(labels)
        grad_numerical_lower /= n_batch
        grad_numerical_upper /= n_batch
        assert_almost_equal(grad_numerical_lower, loss_layer.gradient)
        assert_almost_equal(grad_numerical_upper, upper_layer.gradient)

    def test_gradient_vs_numerical_random_vmax(self):
        np.random.seed(0)
        n_in = 10
        n_upper = 5
        n_lower = 4
        n_batch = 3
        isi = 5e-3
        t_max = 0.2
        loss_pars = VMaxCrossEntropyLossParameters(n=n_lower, n_in=n_lower)
        upper_pars = LIFLayerParameters(
            n=n_upper, n_in=n_in, tau_mem=20e-3, tau_syn=5e-3
        )
        lower_pars = LIFLayerParameters(
            n=n_lower, n_in=n_upper, tau_mem=20e-3, tau_syn=5e-3
        )
        norm_factor = get_normalization_factor(upper_pars.tau_mem, upper_pars.tau_syn)
        w_upper = np.random.normal(0.2, 0.1, size=(n_in, n_upper)) * norm_factor
        w_lower = np.random.normal(0.2, 0.1, size=(n_upper, n_lower)) * norm_factor
        w_vmax = np.random.normal(1, 1, size=(n_lower, n_lower))
        w_eps = 1e-6
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
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                assert lower_layer.post_batch[batch_idx].n_spikes > 0
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_upper)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_minus)
                lower_layer = LIFLayer(lower_pars, w_lower)
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
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
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                loss_plus = loss_layer.get_losses(labels)[batch_idx]

                w_minus = np.copy(w_lower)
                w_minus[idx] -= w_eps
                upper_layer = LIFLayer(upper_pars, w_upper)
                lower_layer = LIFLayer(lower_pars, w_minus)
                loss_layer = VMaxCrossEntropyLoss(loss_pars)
                loss_layer.w_in = w_vmax
                loss_layer(lower_layer(upper_layer(input_spikes)))
                loss_minus = loss_layer.get_losses(labels)[batch_idx]

                grad_numerical_lower[idx] += (loss_plus - loss_minus) / (2 * w_eps)

        upper_layer = LIFLayer(upper_pars, w_upper)
        lower_layer = LIFLayer(lower_pars, w_lower)
        loss_layer = VMaxCrossEntropyLoss(loss_pars)
        loss_layer.w_in = w_vmax
        loss_layer(lower_layer(upper_layer(input_spikes)))
        loss_layer.backward(labels)
        grad_numerical_lower /= n_batch
        grad_numerical_upper /= n_batch
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


if __name__ == "__main__":
    unittest.main()
