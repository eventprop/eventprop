from .layer import SpikeDataset
from .training import AbstractTwoLayer
from .lif_layer import LIFLayer, LIFLayerParameters
from .loss_layer import TTFSCrossEntropyLoss, TTFSCrossEntropyLossParameters


class TTFSMixin:
    def _get_results_for_set(self, dataset: SpikeDataset):
        loss, accuracy = super()._get_results_for_set(dataset)
        first_spikes = self.loss.first_spike_times.copy()
        return loss, accuracy, first_spikes

    def valid(self):
        valid_loss, valid_error, valid_first_spikes = self._get_results_for_set(
            self.valid_batch
        )
        self.valid_accuracies.append(valid_error)
        self.valid_losses.append(valid_loss)
        self.valid_first_spikes.append(valid_first_spikes)
        return valid_loss, valid_error, valid_first_spikes

    def test(self):
        test_loss, test_error, test_first_spikes = self._get_results_for_set(
            self.test_batch
        )
        self.test_accuracies.append(test_error)
        self.test_losses.append(test_loss)
        self.test_first_spikes.append(test_first_spikes)
        return test_loss, test_error, test_first_spikes

    def get_data_for_pickling(self):
        return (
            self.losses,
            self.accuracies,
            self.test_accuracies,
            self.test_losses,
            self.test_first_spikes,
            self.valid_accuracies,
            self.valid_losses,
            self.valid_first_spikes,
            self.weights,
        )

    def reset_results(self):
        super().reset_results()
        self.test_first_spikes, self.valid_first_spikes = list(), list()


class TwoLayerTTFS(TTFSMixin, AbstractTwoLayer):
    def __init__(
        self,
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(),
        **kwargs,
    ):
        super().__init__(
            hidden_layer_class=LIFLayer,
            loss_class=TTFSCrossEntropyLoss,
            hidden_parameters=hidden_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )