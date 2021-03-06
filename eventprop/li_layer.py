import numpy as np
from typing import NamedTuple, List, Iterable

from .layer import Layer, VMax
from .lif_layer_cpp import Spike

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
        self.input_spikes = None
        self.vmax = None
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
        self._exp_input_mem = np.zeros(len(self.input_spikes))
        self._exp_input_syn = np.zeros(len(self.input_spikes))
        for idx, spike in enumerate(self.input_spikes):
            self._exp_input_mem[idx] = np.e ** (spike.time / self.parameters.tau_mem)
            self._exp_input_syn[idx] = np.e ** (spike.time / self.parameters.tau_syn)

    def _compute_sums_for_tmax(self):
        # pre-compute sums that are used in determining maxima
        self.sum0 = np.zeros((len(self.input_spikes) + 1, self.parameters.n))
        self.sum1 = np.zeros((len(self.input_spikes) + 1, self.parameters.n))
        for idx, spike in enumerate(self.input_spikes):
            self.sum0[idx + 1] = (
                self.sum0[idx]
                + self._exp_input_syn[idx] * self.w_in[spike.source_neuron, :]
            )
            self.sum1[idx + 1] = (
                self.sum1[idx]
                + self._exp_input_mem[idx] * self.w_in[spike.source_neuron, :]
            )

    def _tmax(self, input_spike_idx: int, target_nrn_idx: int) -> float:
        """
        Find time of maximum before input spike `input_spike_idx` for neuron `target_nrn_idx`.
        """
        if input_spike_idx == 0:
            return 0
        elif input_spike_idx == len(self.input_spikes):
            # find final maximum (no input spikes after)
            t_input = np.inf
        else:
            t_input = self.input_spikes[input_spike_idx].time
        # get pre-computed sum
        sum_0 = self.sum0[input_spike_idx, target_nrn_idx]
        sum_1 = self.sum1[input_spike_idx, target_nrn_idx]
        largest_t_pre = self.input_spikes[input_spike_idx - 1].time
        # compute solution of dv/dt=0
        tmax = self._tmax_prefactor * (self._tmax_summand + np.log(sum_1 / sum_0))
        if largest_t_pre <= tmax <= t_input:
            return tmax
        return None

    def _v(self, t: float, target_nrn_idx: int) -> float:
        times = np.array([s.time for s in self.input_spikes if s.time <= t])
        sources = [s.source_neuron for s in self.input_spikes if s.time <= t]
        kernels = self._k(t - times)
        v = sum(
            [
                self.w_in[source, target_nrn_idx] * k
                for source, k in zip(sources, kernels)
            ]
        )
        return v

    def forward(self, input_spikes: List[Spike], code: str = "python") -> np.ndarray:
        super().forward(input_spikes)
        assert all([x.source_neuron < self.parameters.n_in for x in input_spikes])
        self.input_spikes.sort(key=lambda x: x.time)
        if code == "python":
            self.compute_vmax_python(input_spikes)
            return self.vmax
        elif code == "cpp":
            raise NotImplementedError()
        raise RuntimeError(f"Code not recognized: {code}.")

    def compute_vmax_python(self, input_spikes: List[Spike]):
        self._compute_exponentiated_times()
        self._compute_sums_for_tmax()
        self.vmax = list()
        for target_nrn_idx in range(self.parameters.n):
            self.vmax.append(self.get_vmax_for_neuron(target_nrn_idx))
        self._ran_forward = True

    def backward(self, code: str = "python"):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        if code == "cpp":
            raise NotImplementedError()
        else:
            nrn_events = self.input_spikes
            if len(nrn_events) == 0:
                return
            # iterate pre spikes in reverse order
            nrn_events.sort(key=lambda x: x.time, reverse=True)
            largest_time = nrn_events[0].time
            for event in nrn_events:
                t_bwd = largest_time - event.time
                # pre spike -> gradient sample
                for idx, vmax in enumerate(self.vmax):
                    if vmax.time is None:
                        continue
                    t_vmax_bwd = largest_time - vmax.time
                    if t_vmax_bwd > t_bwd:
                        continue
                    lambda_v = (
                        -np.exp(-(t_bwd - t_vmax_bwd) / self.parameters.tau_mem)
                        * vmax.error
                        / self.parameters.tau_mem
                    )
                    lambda_i = (
                        -self._k_bwd(t_bwd - t_vmax_bwd)
                        * vmax.error
                        / self.parameters.tau_mem
                    )
                    self.gradient[event.source_neuron, idx] += (
                        -self.parameters.tau_syn * lambda_i
                    )
                    outbound_signal = self.w_in[event.source_neuron, idx] * (
                        lambda_v - lambda_i
                    )
                    if event.source_layer is not None:
                        event.error += outbound_signal
        self._ran_backward = True
        super().backward()

    def get_vmax_for_neuron(self, target_nrn_idx: int):
        if self.w_in is None:
            raise RuntimeError("Set weights first!")
        finished = False
        final_spike = Spike(time=np.finfo(np.float64).max, source_neuron=-1)
        processed_up_to = 0
        total_vmax = VMax(None, self.parameters.v_leak, 0)
        while not finished:
            for i, spike in enumerate(self.input_spikes + [final_spike]):
                if i < processed_up_to:
                    continue
                x = self._tmax(i, target_nrn_idx)
                if x is not None:
                    # check for post spike
                    vmax = self._v(x, target_nrn_idx)
                    if vmax > total_vmax.value:
                        total_vmax.time = x
                        total_vmax.value = vmax
                vmax = self._v(spike.time, target_nrn_idx)
                if vmax > total_vmax.value:
                    total_vmax.time = x
                    total_vmax.value = vmax
                if i == len(self.input_spikes):
                    finished = True
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
        self.gradient = np.zeros_like(self.gradient)
