from typing import NamedTuple, List
import numpy as np

from .layer import Layer, Spikes
from .li_layer import LILayer, LILayerParameters

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

    def forward(self, input_spikes: Spikes):
        super().forward(input_spikes)
        self._ran_forward = True

    def _find_first_spikes(self):
        self.first_spike_times = np.full(self.parameters.n, np.inf)
        self.first_spike_idxs = [None] * self.parameters.n
        for neuron in range(self.parameters.n):
            idxs = np.argwhere(self.input_spikes.sources == neuron)
            if len(idxs) > 0:
                self.first_spike_times[neuron] = self.input_spikes.times[idxs[0, 0]]
                self.first_spike_idxs[neuron] = idxs[0, 0]

    def get_loss(self, correct_label_neuron: int):
        """
        Compute cross-entropy loss over first spike times
        """
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        # Find first spike times
        self._find_first_spikes()
        if np.isposinf(self.first_spike_times[correct_label_neuron]):
            return np.nan
        t_label = self.first_spike_times[correct_label_neuron]

        loss = 0
        sum0 = np.sum(np.exp(-self.first_spike_times / self.parameters.tau0))
        loss += -np.log(np.exp(-t_label / self.parameters.tau0) / sum0)
        loss += self.parameters.alpha * (np.exp(t_label / self.parameters.tau1) - 1)
        return loss

    def get_classification_result(self, correct_label_neuron: int):
        self._find_first_spikes()
        if np.isposinf(self.first_spike_times[correct_label_neuron]):
            return 0  # wrong classification
        label_time = self.first_spike_times[correct_label_neuron]
        if np.all(self.first_spike_times >= label_time):
            return 1
        else:
            return 0

    def backward(self, correct_label_neuron: int):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        self._find_first_spikes()
        self.input_spikes.errors[:] = 0
        if np.isposinf(self.first_spike_times[correct_label_neuron]):
            return  # no spike, no error, no backprop
        tau0, tau1, alpha = (
            self.parameters.tau0,
            self.parameters.tau1,
            self.parameters.alpha,
        )
        sum0 = np.sum(np.exp(-self.first_spike_times / tau0))
        # compute error for label neuron first
        label_error = 0
        t_label = self.first_spike_times[correct_label_neuron]
        exp_t_label = np.exp(-t_label / tau0)
        exp_t_label_squared = np.exp(-2 * t_label / tau0)
        label_error += -(sum0 / exp_t_label) * (
            -exp_t_label / (tau0 * sum0) + exp_t_label_squared / (tau0 * sum0 ** 2)
        )
        label_error += alpha / tau1 * np.exp(t_label / tau1)
        self.input_spikes.errors[
            self.first_spike_idxs[correct_label_neuron]
        ] = label_error
        # compute errors for other neurons
        for nrn_idx in range(self.parameters.n):
            if nrn_idx == correct_label_neuron or np.isposinf(
                self.first_spike_times[nrn_idx]
            ):
                continue
            error = 0
            t = self.first_spike_times[nrn_idx]
            error += -1 / (tau0 * sum0) * np.exp(-t / tau0)
            self.input_spikes.errors[self.first_spike_idxs[nrn_idx]] = error
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
