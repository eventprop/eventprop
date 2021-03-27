from eventprop.loss_layer import VMaxCrossEntropyLossParameters
from eventprop.vmax_training import OneLayerVMax
import numpy as np
import os

from eventprop.layer import GaussianDistribution, SpikeDataset, DiagonalWeights
from eventprop.eventprop_cpp import Spikes, SpikesVector
from eventprop.lif_layer import LIFLayerParameters
from eventprop.ttfs_training import TwoLayerTTFS, TTFSCrossEntropyLossParameters
from eventprop.optimizer import GradientDescentParameters

dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "yin_yang_data_set/publication_data"
)


class YinYangMixin:
    t_min: float = 10e-3
    t_max: float = 40e-3
    t_bias: float = 20e-3

    def load_data(self):
        train_samples = np.load(os.path.join(dir_path, "train_samples.npy"))
        test_samples = np.load(os.path.join(dir_path, "test_samples.npy"))
        valid_samples = np.load(os.path.join(dir_path, "validation_samples.npy"))
        train_labels = np.load(os.path.join(dir_path, "train_labels.npy"))
        test_labels = np.load(os.path.join(dir_path, "test_labels.npy"))
        valid_labels = np.load(os.path.join(dir_path, "validation_labels.npy"))

        def get_batch(samples, labels):
            spikes = list()
            for s in samples:
                times = np.array(
                    [self.t_min + x * (self.t_max - self.t_min) for x in s]
                    + [self.t_bias],
                    dtype=np.float64,
                )
                sources = np.arange(len(s) + 1, dtype=np.int32)
                sort_idxs = np.argsort(times)
                times = times[sort_idxs]
                sources = sources[sort_idxs]
                spikes.append(Spikes(times, sources))
            return SpikeDataset(SpikesVector(spikes), labels)

        self.train_batch, self.test_batch, self.valid_batch = (
            get_batch(train_samples, train_labels),
            get_batch(test_samples, test_labels),
            get_batch(valid_samples, valid_labels),
        )


class YinYangTTFS(YinYangMixin, TwoLayerTTFS):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(
            minibatch_size=200, iterations=30000, lr=1e-3, gradient_clip=None
        ),
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=5,
            n=200,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(w_mean=2, w_std=1),
        ),
        output_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=200,
            n=3,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(w_mean=0.4, w_std=0.4),
        ),
        loss_parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(
            n=3
        ),
        **kwargs,
    ):
        super().__init__(
            gd_parameters=gd_parameters,
            hidden_parameters=hidden_parameters,
            output_parameters=output_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )


class YinYangVMax(YinYangMixin, OneLayerVMax):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(
            minibatch_size=200, iterations=30000, lr=1e-3, gradient_clip=None
        ),
        output_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=5,
            n=200,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(w_mean=2, w_std=1),
        ),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(
            n=3, n_in=200, w_dist=GaussianDistribution(w_mean=0.5, w_std=0.5)
        ),
        **kwargs,
    ):
        super().__init__(
            gd_parameters=gd_parameters,
            output_parameters=output_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )

