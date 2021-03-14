import numpy as np
from typing import NamedTuple, List, Iterable

from .layer import Layer
from .eventprop_cpp import (
    Spikes,
    Maxima,
    MaximaVector,
    SpikesVector,
    compute_maxima_batch_cpp,
    backward_maxima_batch_cpp,
)

# fmt: off
class LILayerParameters(NamedTuple):
    n           : int   = 10
    n_in        : int   = 10
    tau_mem     : float = 20e-3 # s
    tau_syn     : float = 5e-3  # s
    v_leak      : float = 0
    w_mean      : float = None
    w_std       : float = None
# fmt: on


class LILayer(Layer):
    def __init__(self, parameters: LILayerParameters, w_in: np.array = None):
        super().__init__()
        self.parameters = parameters
        if w_in is not None:
            assert isinstance(w_in, np.ndarray)
            assert w_in.shape == (self.parameters.n_in, self.parameters.n)
            self.w_in = w_in
        elif self.parameters.w_mean is not None and self.parameters.w_std is not None:
            self.w_in = np.random.normal(
                self.parameters.w_mean,
                self.parameters.w_std,
                size=(self.parameters.n_in, self.parameters.n),
            )
        else:
            self.w_in = None
        self.input_batch = None
        self.gradient = np.zeros_like(self.w_in)
        self._k_prefactor = self.parameters.tau_syn / (
            self.parameters.tau_mem - self.parameters.tau_syn
        )
        self._k_bwd_prefactor = self.parameters.tau_mem / self.parameters.tau_syn
        self._tmax_prefactor = 1 / (
            1 / self.parameters.tau_mem - 1 / self.parameters.tau_syn
        )
        self._tmax_summand = np.log(self.parameters.tau_syn / self.parameters.tau_mem)
        self._exp_input_syn = None
        self._exp_input_mem = None
        self._exp_mem_prefactor = np.e ** (-1 / self.parameters.tau_mem)
        self._exp_syn_prefactor = np.e ** (-1 / self.parameters.tau_syn)

    # solution of tau_syn * f' = -f', tau_mem * g' = f-g
    def _k(self, t: float) -> float:
        return self._k_prefactor * (
            np.exp(-t / self.parameters.tau_mem) - np.exp(-t / self.parameters.tau_syn)
        )

    def _k_bwd(self, t: float) -> float:
        return self._k_bwd_prefactor * self._k(t)

    def _compute_exponentiated_times(self):
        self._exp_input_mem = np.exp(self.input_spikes.times / self.parameters.tau_mem)
        self._exp_input_syn = np.exp(self.input_spikes.times / self.parameters.tau_syn)

    def _allocate_sums_for_tmax(self):
        self.sum0 = np.zeros((len(self.input_spikes.times) + 1, self.parameters.n))
        self.sum1 = np.zeros((len(self.input_spikes.times) + 1, self.parameters.n))

    def _compute_sums_for_tmax(self):
        # pre-compute sums that are used in determining maxima
        self._allocate_sums_for_tmax()
        self.sum0[1:, :] = np.cumsum(
            self._exp_input_syn[:, None] * self.w_in[self.input_spikes.sources], axis=0
        )
        self.sum1[1:, :] = np.cumsum(
            self._exp_input_mem[:, None] * self.w_in[self.input_spikes.sources], axis=0
        )

    def _tmax(self, input_spike_idx: int, target_nrn_idx: int) -> float:
        """
        Find time of maximum before input spike `input_spike_idx` for neuron `target_nrn_idx`.
        """
        if input_spike_idx == 0:
            return 0
        elif input_spike_idx == self.input_spikes.n_spikes:
            # find final maximum (no input spikes after)
            t_input = np.inf
        else:
            t_input = self.input_spikes.times[input_spike_idx]
        # get pre-computed sum
        sum_0 = self.sum0[input_spike_idx, target_nrn_idx]
        sum_1 = self.sum1[input_spike_idx, target_nrn_idx]
        largest_t_pre = self.input_spikes.times[input_spike_idx - 1]
        # compute solution of dv/dt=0
        tmax = self._tmax_prefactor * (self._tmax_summand + np.log(sum_1 / sum_0))
        if largest_t_pre <= tmax <= t_input:
            return tmax
        return None

    def _v(self, t: float, target_nrn_idx: int, t_pre_idx=None) -> float:
        if t_pre_idx is None:
            idxs = np.argwhere(self.input_spikes.times <= t)
            if len(idxs) == 0:
                return 0
            else:
                t_pre_idx = idxs[-1, 0] + 1
        v = self._k_prefactor * (
            self.sum1[t_pre_idx, target_nrn_idx]
            * np.e ** (-t / self.parameters.tau_mem)
            - self.sum0[t_pre_idx, target_nrn_idx]
            * np.e ** (-t / self.parameters.tau_syn)
        )
        return v

    def forward(self, input_batch: SpikesVector):
        super().forward(input_batch)
        self.maxima_batch = compute_maxima_batch_cpp(
            self.w_in, input_batch, self.parameters.tau_mem, self.parameters.tau_syn
        )
        self._ran_forward = True
        return self.maxima_batch

    def backward(self):
        backward_maxima_batch_cpp(
            self.input_batch,
            self.maxima_batch,
            self.w_in,
            self.gradient,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )
        self._ran_backward = True
        super().backward()

    # def forward(self, input_spikes: Spikes, code: str = "python") -> np.ndarray:
    #    super().forward(input_spikes)
    #    if code == "python":
    #        self.compute_vmax_python(input_spikes)
    #        return self.vmax
    #    elif code == "cpp":
    #        raise NotImplementedError()
    #    raise RuntimeError(f"Code not recognized: {code}.")

    def compute_vmax_python(self, input_spikes: Spikes):
        self._compute_exponentiated_times()
        self._compute_sums_for_tmax()
        self.vmax = list()
        for target_nrn_idx in range(self.parameters.n):
            self.vmax.append(self.get_vmax_for_neuron(target_nrn_idx))
        self._ran_forward = True

    # def backward(self, code: str = "python"):
    #    if not self._ran_forward:
    #        raise RuntimeError("Run forward first!")
    #    if code == "cpp":
    #        raise NotImplementedError()
    #    else:
    #        # iterate pre spikes in reverse order
    #        largest_time = self.input_spikes.times[-1]
    #        all_spike_sort_idxs = np.arange(self.input_spikes.n_spikes)[::-1]
    #        for spike_idx in all_spike_sort_idxs:
    #            t_bwd = largest_time - self.input_spikes.times[spike_idx]
    #            # pre spike -> gradient sample
    #            for idx, vmax in enumerate(self.vmax):
    #                if vmax.time is None:
    #                    continue
    #                t_vmax_bwd = largest_time - vmax.time
    #                if t_vmax_bwd > t_bwd:
    #                    continue
    #                lambda_v = (
    #                    -np.exp(-(t_bwd - t_vmax_bwd) / self.parameters.tau_mem)
    #                    * vmax.error
    #                    / self.parameters.tau_mem
    #                )
    #                lambda_i = (
    #                    -self._k_bwd(t_bwd - t_vmax_bwd)
    #                    * vmax.error
    #                    / self.parameters.tau_mem
    #                )
    #                self.gradient[self.input_spikes.sources[spike_idx], idx] += (
    #                    -self.parameters.tau_syn * lambda_i
    #                )
    #                outbound_signal = self.w_in[
    #                    self.input_spikes.sources[spike_idx], idx
    #                ] * (lambda_v - lambda_i)
    #                self.input_spikes.errors[spike_idx] += outbound_signal
    #    self._ran_backward = True
    #    super().backward()

    def get_vmax_for_neuron(self, target_nrn_idx: int):
        if self.w_in is None:
            raise RuntimeError("Set weights first!")
        total_vmax = VMax(None, self.parameters.v_leak, 0)
        for spike_idx in range(0, self.input_spikes.n_spikes + 1):
            if spike_idx == self.input_spikes.n_spikes:
                spike_time = np.inf
            else:
                spike_time = self.input_spikes.times[spike_idx]
            x = self._tmax(spike_idx, target_nrn_idx)
            if x is not None:
                vmax = self._v(x, target_nrn_idx, spike_idx)
                if vmax > total_vmax.value:
                    total_vmax.time = x
                    total_vmax.value = vmax
            vmax = self._v(spike_time, target_nrn_idx, spike_idx)
            if vmax > total_vmax.value:
                total_vmax.time = spike_time
                total_vmax.value = vmax
        return total_vmax

    def get_voltage_trace_for_neuron(
        self,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
        code: str = "python",
    ) -> np.array:
        if self.w_in is None:
            raise RuntimeError("Set weights first!")
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        ts = np.arange(0, t_max, step=dt)
        if code == "python":
            v = np.zeros_like(ts)
            for t_idx, t in enumerate(ts):
                v[t_idx] = self._v(t, target_nrn_idx)
            return ts, v
        elif code == "cpp":
            raise NotImplementedError()

    def zero_grad(self):
        self.gradient[:] = 0
