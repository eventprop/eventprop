from typing import NamedTuple, List
import numpy as np

from .layer import Layer
from .li_layer import LILayer, LILayerParameters
from .lif_layer_cpp import SpikesVector

# fmt: off
class TTFSCrossEntropyLossParameters(NamedTuple):
    n           : int   = 10
    alpha       : float = 1e-2
    tau0        : float = 2e-3  # s
    tau1        : float = 10e-3 # s
# fmt: on

VMaxCrossEntropyLossParameters = LILayerParameters


class TTFSCrossEntropyLoss(Layer):
    def __init__(
        self,
        parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(),
    ):
        super().__init__()
        self.parameters = parameters
        self.first_spike_times = None
        self.first_spike_idxs = None

    def forward(self, input_batch: SpikesVector):
        super().forward(input_batch)
        self._batch_idxs = np.arange(len(self.input_batch))
        # Find first spike times
        self._find_first_spikes()
        self._ran_forward = True

    def _find_first_spikes(self):
        self.first_spike_times = np.full(
            (len(self.input_batch), self.parameters.n), np.nan
        )
        self.first_spike_idxs = np.empty(
            (len(self.input_batch), self.parameters.n), dtype=np.int
        )
        for batch_idx, spikes in enumerate(self.input_batch):
            for neuron in range(self.parameters.n):
                idxs = np.argwhere(spikes.sources == neuron)
                if len(idxs) > 0:
                    self.first_spike_times[batch_idx, neuron] = spikes.times[idxs[0, 0]]
                    self.first_spike_idxs[batch_idx, neuron] = idxs[0, 0]

    def get_losses(self, labels: np.ndarray):
        """
        Compute cross-entropy losses over first spike times
        """
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        t_labels = self.first_spike_times[self._batch_idxs, labels]
        sum0 = np.nansum(np.exp(-self.first_spike_times / self.parameters.tau0), axis=1)
        loss = -np.log(np.exp(-t_labels / self.parameters.tau0) / sum0)
        loss += self.parameters.alpha * (np.exp(t_labels / self.parameters.tau1) - 1)
        return loss

    def get_accuracy(self, labels: np.ndarray):
        t_labels = self.first_spike_times[self._batch_idxs, labels]
        return np.mean(
            np.all(
                np.logical_or(
                    self.first_spike_times >= t_labels[:, None],
                    np.isnan(self.first_spike_times),
                ),
                axis=1,
            )
        )

    def backward(self, labels: np.ndarray):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        tau0, tau1, alpha = (
            self.parameters.tau0,
            self.parameters.tau1,
            self.parameters.alpha,
        )
        sum0 = np.nansum(np.exp(-self.first_spike_times / tau0), axis=1)
        # compute error for label neuron first
        t_labels = self.first_spike_times[self._batch_idxs, labels]
        exp_t_label = np.exp(-t_labels / tau0)
        exp_t_label_squared = np.exp(-2 * t_labels / tau0)
        label_error = -(1 / exp_t_label) * (
            -exp_t_label / tau0 + exp_t_label_squared / (tau0 * sum0)
        )
        label_error += alpha / tau1 * np.exp(t_labels / tau1)
        for batch_idx in range(len(self.input_batch)):
            if not np.isnan(label_error[batch_idx]):
                self.input_batch[batch_idx].set_error(
                    self.first_spike_idxs[batch_idx, labels[batch_idx]],
                    label_error[batch_idx],
                )
        # compute errors for other neurons
        errors = -1 / (tau0 * sum0[:, None]) * np.exp(-self.first_spike_times / tau0)
        for batch_idx in range(len(self.input_batch)):
            if np.isnan(label_error[batch_idx]):
                continue
            for nrn_idx in range(self.parameters.n):
                if nrn_idx == labels[batch_idx]:
                    continue
                if not np.isnan(errors[batch_idx, nrn_idx]):
                    self.input_batch[batch_idx].set_error(
                        self.first_spike_idxs[batch_idx, nrn_idx],
                        errors[batch_idx, nrn_idx],
                    )
        self._ran_backward = True
        super().backward()


class VMaxCrossEntropyLoss(LILayer):
    def get_loss(self, correct_label_neuron: int):
        """
        Compute cross-entropy loss over voltage maxima
        """
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        v_max_label = self.vmax[correct_label_neuron].value
        loss = -np.log(
            np.exp(v_max_label) / np.sum([np.exp(vmax.value) for vmax in self.vmax])
        )
        return loss

    def get_classification_result(self, correct_label_neuron: int):
        v_max_label = self.vmax[correct_label_neuron].value
        if all([v_max_label >= vmax.value for vmax in self.vmax]):
            return 1
        else:
            return 0

    def backward(self, correct_label_neuron: int, code: str = "python"):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")

        exp_sum = np.sum([np.exp(vmax.value) for vmax in self.vmax])
        label_error = np.exp(self.vmax[correct_label_neuron].value) / exp_sum - 1
        self.vmax[correct_label_neuron].error = label_error

        for nrn_idx in range(self.parameters.n):
            if nrn_idx == correct_label_neuron:
                continue
            error = np.exp(self.vmax[nrn_idx].value) / exp_sum
            self.vmax[nrn_idx].error = error
        self._ran_backward = True
        super().backward(code=code)
