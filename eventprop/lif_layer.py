import numpy as np
from scipy.optimize import brentq
import logging
from typing import NamedTuple

from .layer import Layer, Spikes
from .lif_layer_cpp import compute_spikes_cpp, backward_cpp
from . import lif_layer_cpp


# fmt: off
class LIFLayerParameters(NamedTuple):
    n             : int   = 10
    n_in          : int   = 10
    tau_mem       : float = 20e-3 # s
    tau_syn       : float = 5e-3  # s
    v_th          : float = 1.
    v_leak        : float = 0
    w_mean        : float = None
    w_std         : float = None
    n_spikes_max  : int   = 100
# fmt: on


class LIFLayer(Layer):
    def __init__(self, parameters: LIFLayerParameters, w_in: np.array = None):
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
        self.post_spikes = Spikes(np.array([]), np.array([]), source_layer=id(self))
        self.input_spikes = None
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
        self._precomputed_id = None

    def _reset_post_spikes_per_neuron(self):
        self._post_spikes_per_neuron = [
            Spikes(
                np.full(self.parameters.n_spikes_max, np.nan),
                np.full(self.parameters.n_spikes_max, nrn_idx),
                n_spikes=0,
            )
            for nrn_idx in range(self.parameters.n)
        ]

    def _concatenate_post_spikes(self):
        all_times = np.concatenate(
            [x.times[: x.n_spikes] for x in self._post_spikes_per_neuron]
        )
        all_sources = np.concatenate(
            [x.sources[: x.n_spikes] for x in self._post_spikes_per_neuron]
        )
        sort_mask = np.argsort(all_times)
        self.post_spikes.times = all_times[sort_mask]
        self.post_spikes.sources = all_sources[sort_mask]
        self.post_spikes.n_spikes = len(self.post_spikes.times)
        self.post_spikes.errors = np.zeros_like(self.post_spikes.times)

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
        # get pre-computed sums
        sum_0 = self.sum0[input_spike_idx, target_nrn_idx]
        sum_1 = self.sum1[input_spike_idx, target_nrn_idx]
        largest_t_pre = self.input_spikes.times[input_spike_idx - 1]
        # add terms that are due to post spikes
        post_times = self._post_spikes_per_neuron[target_nrn_idx].times
        post_mask = post_times <= t_input
        sum_1 -= np.sum(
            self.parameters.v_th
            * np.exp(post_times[post_mask] / self.parameters.tau_mem)
        )
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
        post_times = self._post_spikes_per_neuron[target_nrn_idx].times
        post_mask = post_times <= t
        v -= np.sum(
            self.parameters.v_th
            * np.exp(-(t - post_times[post_mask]) / self.parameters.tau_mem)
        )
        return v

    def forward(self, input_spikes: Spikes, code: str = "cpp") -> Spikes:
        super().forward(input_spikes)
        if code == "python":
            self.compute_spikes_python()
        elif code == "cpp":
            times, sources = compute_spikes_cpp(
                self.w_in,
                self.input_spikes.times,
                self.input_spikes.sources,
                self.parameters.v_th,
                self.parameters.tau_mem,
                self.parameters.tau_syn,
            )
            self.post_spikes.times = times
            self.post_spikes.sources = sources
            self.post_spikes.n_spikes = self.post_spikes.times.size
            self.post_spikes.errors = np.zeros(
                self.post_spikes.n_spikes, dtype=np.float64
            )
        else:
            raise RuntimeError(f"Code not recognized: {code}.")
        self._ran_forward = True
        return self.post_spikes

    def compute_spikes_python(self):
        self._reset_post_spikes_per_neuron()
        if not self._precomputed_id == id(self.input_spikes.times):
            if self.input_spikes.n_spikes == 0:
                self._exp_input_syn = np.array([])
                self._exp_input_mem = np.array([])
            else:
                self._compute_exponentiated_times()
            self._compute_sums_for_tmax()
            self._precomputed_id = id(self.input_spikes.times)
        for target_nrn_idx in range(self.parameters.n):
            self.compute_spikes_for_neuron(target_nrn_idx)
        self._concatenate_post_spikes()
        return self.post_spikes

    def _bracket_spike(self, t_before, t_after, target_nrn_idx, spike_idx):
        return brentq(
            lambda t: self._v(t, target_nrn_idx, spike_idx) - 1,
            t_before,
            t_after,
            disp=True,
        )

    def backward(self, code: str = "cpp"):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        if self.post_spikes.n_spikes == 0:
            self._ran_backward = True
            return  # no backprop if no spikes
        # FIXME
        if code == "cpp":
            backward_cpp(
                self.input_spikes.times,
                self.input_spikes.sources,
                self.post_spikes.times,
                self.post_spikes.sources,
                self.input_spikes.errors,
                self.post_spikes.errors,
                self.w_in,
                self.gradient,
                self.parameters.v_th,
                self.parameters.tau_mem,
                self.parameters.tau_syn,
            )
        elif code == "python":
            lambda_v = np.zeros(self.parameters.n)
            lambda_i_jumps = np.full_like(self.post_spikes.times, np.nan)
            lambda_i_sources = np.full_like(self.post_spikes.times, np.nan)
            lambda_i_times = np.full_like(self.post_spikes.times, np.nan)
            n_lambda_i_jumps = 0
            # iterate pre, post spikes in reverse order
            input_spike_ids = np.arange(self.input_spikes.n_spikes)
            post_spike_ids = np.arange(self.post_spikes.n_spikes)
            spike_types = np.concatenate(
                [
                    np.ones_like(self.input_spikes.times),
                    np.zeros_like(self.post_spikes.times),
                ]
            )
            all_spike_sources = np.concatenate(
                [self.input_spikes.sources, self.post_spikes.sources]
            )
            all_spike_idxs = np.concatenate([input_spike_ids, post_spike_ids])
            all_spike_times = np.concatenate(
                [self.input_spikes.times, self.post_spikes.times]
            )
            all_spike_sort_idxs = np.argsort(all_spike_times)[::-1]

            largest_time = all_spike_times[all_spike_sort_idxs[0]]
            previous_t = -np.inf
            is_pre_spike = lambda idx: spike_types[idx] == 1
            for idx in all_spike_sort_idxs:
                spike_time = all_spike_times[idx]
                spike_source = all_spike_sources[idx]
                t_bwd = largest_time - spike_time
                # pre spike -> gradient sample
                if is_pre_spike(idx):
                    lambda_v = (
                        np.exp(-(t_bwd - previous_t) / self.parameters.tau_mem)
                        * lambda_v
                    )
                    lambda_i = np.zeros_like(lambda_v)
                    for nrn_idx in range(self.parameters.n):
                        lambda_i_mask = lambda_i_sources == nrn_idx
                        lambda_i[nrn_idx] = np.sum(
                            lambda_i_jumps[lambda_i_mask]
                            * self._k_bwd(t_bwd - lambda_i_times[lambda_i_mask])
                        )
                    self.gradient[spike_source, :] += (
                        -self.parameters.tau_syn * lambda_i
                    )
                    outbound_signal = np.dot(
                        self.w_in[spike_source, :], lambda_v - lambda_i
                    )
                    self.input_spikes.errors[all_spike_idxs[idx]] += outbound_signal
                # post spike -> jump and absorb error
                else:
                    i = self._i(spike_time, spike_source)
                    # decay down first
                    lambda_v = (
                        np.exp(-(t_bwd - previous_t) / self.parameters.tau_mem)
                        * lambda_v
                    )
                    lambda_i_jump = (
                        1
                        / (i - self.parameters.v_th)
                        * (
                            self.parameters.v_th * lambda_v[spike_source]
                            + self.post_spikes.errors[all_spike_idxs[idx]]
                        )
                    )
                    lambda_i_sources[n_lambda_i_jumps] = spike_source
                    lambda_i_times[n_lambda_i_jumps] = t_bwd
                    lambda_i_jumps[n_lambda_i_jumps] = lambda_i_jump
                    n_lambda_i_jumps += 1
                    # self-jump and error absorption
                    lambda_v[spike_source] = (
                        i / (i - self.parameters.v_th) * lambda_v[spike_source]
                        + 1
                        / (i - self.parameters.v_th)
                        * self.post_spikes.errors[all_spike_idxs[idx]]
                    )
                previous_t = t_bwd
        self._ran_backward = True
        super().backward()

    def _resize_post_spikes(self, target_nrn_idx: int):
        self._post_spikes_per_neuron[target_nrn_idx].times = np.concatenate(
            [
                self._post_spikes_per_neuron[target_nrn_idx].times,
                np.full(self.parameters.n_spikes_max, np.nan),
            ]
        )
        self._post_spikes_per_neuron[target_nrn_idx].sources = np.concatenate(
            [
                self._post_spikes_per_neuron[target_nrn_idx].sources,
                np.full(self.parameters.n_spikes_max, target_nrn_idx),
            ]
        )
        self._post_spikes_per_neuron[target_nrn_idx].errors = np.concatenate(
            [
                self._post_spikes_per_neuron[target_nrn_idx].errors,
                np.full(self.parameters.n_spikes_max, 0),
            ]
        )

    def compute_spikes_for_neuron(self, target_nrn_idx: int):
        if self.w_in is None:
            raise RuntimeError("Set weights first!")
        finished = False
        post_times = self._post_spikes_per_neuron[target_nrn_idx].times
        n_post_spikes = 0
        processed_up_to = 0
        while not finished:
            for spike_idx in range(processed_up_to, self.input_spikes.n_spikes + 1):
                if spike_idx == self.input_spikes.n_spikes:
                    spike_time = np.inf
                else:
                    spike_time = self.input_spikes.times[spike_idx]
                x = self._tmax(spike_idx, target_nrn_idx)
                if x is not None:
                    # check for post spike
                    vmax = self._v(x, target_nrn_idx, spike_idx)
                    if vmax > self.parameters.v_th + np.finfo(float).eps:
                        if n_post_spikes + 1 >= len(post_times):
                            logging.warn("Resizing array to find more spikes.")
                            self._resize_post_spikes(target_nrn_idx)
                            post_times = self._post_spikes_per_neuron[
                                target_nrn_idx
                            ].times
                        if spike_idx == 0:
                            t_before = 0
                        else:
                            t_before = self.input_spikes.times[spike_idx - 1]
                        # Find earliest post spike before t_pre
                        if n_post_spikes > 0:
                            post_mask = post_times < spike_time
                            last_post = post_times[post_mask][-1]
                            if last_post > t_before:
                                t_before = last_post
                        t0 = self._bracket_spike(t_before, x, target_nrn_idx, spike_idx)
                        post_times[n_post_spikes] = t0
                        n_post_spikes += 1
                        processed_up_to = spike_idx
                        break
                if (
                    self._v(spike_time, target_nrn_idx, spike_idx)
                    > self.parameters.v_th + np.finfo(float).eps
                ):
                    if n_post_spikes + 1 >= len(post_times):
                        logging.warn("Resizing array to find more spikes.")
                        self._resize_post_spikes(target_nrn_idx)
                        post_times = self._post_spikes_per_neuron[target_nrn_idx].times
                    t_before = self.input_spikes.times[spike_idx - 1]
                    x = spike_time
                    t0 = self._bracket_spike(t_before, x, target_nrn_idx, spike_idx)
                    post_times[n_post_spikes] = t0
                    n_post_spikes += 1
                    processed_up_to = spike_idx
                    break
                if spike_idx == self.input_spikes.n_spikes:
                    finished = True
        self._post_spikes_per_neuron[target_nrn_idx].n_spikes = n_post_spikes

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

    def get_lambda_i_trace_for_neuron(
        self,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
        code: str = "cpp",
    ) -> np.array:
        if code == "python":
            raise NotImplementedError()
        elif code == "cpp":
            trace = self._lif_cpp.get_lambda_i_trace(target_nrn_idx, t_max, dt=dt)
            ts = np.linspace(0, t_max, len(trace))
            return ts, trace

    def get_lambda_i_for_neuron(
        self, target_nrn_idx: int, t: float, code: str = "cpp"
    ) -> float:
        if code == "python":
            raise NotImplementedError()
        elif code == "cpp":
            return self._lif_cpp.lambda_i(t, target_nrn_idx)

    def _i(self, t: float, target_nrn_idx: int) -> float:
        idxs = np.argwhere(self.input_spikes.times <= t)
        if len(idxs) == 0:
            return 0
        else:
            t_pre_idx = idxs[-1, 0] + 1
        i = np.exp(-t / self.parameters.tau_syn) * self.sum0[t_pre_idx, target_nrn_idx]
        return i
