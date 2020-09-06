import numpy as np
from scipy.optimize import brentq
import logging
from typing import NamedTuple, List, Iterable

from .layer import Layer
from eventprop.lif_layer_cpp import get_spikes, Spike, LIF

class LIFLayerParameters(NamedTuple):
    n           : int   = 10
    n_in        : int   = 10
    tau_mem     : float = 20e-3 # s
    tau_syn     : float = 5e-3  # s
    v_th        : float = 1.
    v_leak      : float = 0
    w_mean      : float = None
    w_std       : float = None

class LIFLayer(Layer):
    def __init__(self, parameters : LIFLayerParameters, w_in : np.array = None):
        super().__init__()
        self.parameters = parameters
        if w_in is not None:
            assert(isinstance(w_in, np.ndarray))
            assert(w_in.shape == (self.parameters.n_in, self.parameters.n))
            self.w_in = w_in
        elif self.parameters.w_mean is not None and self.parameters.w_std is not None:
            self.w_in = np.random.normal(self.parameters.w_mean, self.parameters.w_std, size=(self.parameters.n_in, self.parameters.n))
        else:
            self.w_in = None
        self.post_spikes = list()
        self.input_spikes = None
        self.gradient = np.zeros_like(self.w_in)
        self._k_prefactor = self.parameters.tau_syn/(self.parameters.tau_mem - self.parameters.tau_syn)
        self._k_bwd_prefactor = (self.parameters.tau_mem/self.parameters.tau_syn)
        self._tmax_prefactor = 1/(1/self.parameters.tau_mem - 1/self.parameters.tau_syn)
        self._tmax_summand = np.log(self.parameters.tau_syn/self.parameters.tau_mem)
        self._exp_input_syn = None
        self._exp_input_mem = None
        self._exp_mem_prefactor = np.e**(-1/self.parameters.tau_mem)
        self._exp_syn_prefactor = np.e**(-1/self.parameters.tau_syn)
        self._post_spikes_per_neuron = [list() for _ in range(self.parameters.n)]
        self._lif_cpp = LIF(id(self), self.parameters.v_th, self.parameters.tau_mem, self.parameters.tau_syn, self.w_in)

    # solution of tau_syn * f' = -f', tau_mem * g' = f-g
    def _k(self, t : float) -> float:
        return self._k_prefactor * (np.exp(-t/self.parameters.tau_mem) - np.exp(-t/self.parameters.tau_syn))

    def _k_bwd(self, t : float) -> float:
        return  self._k_bwd_prefactor * self._k(t)

    def _compute_exponentiated_times(self):
        self._exp_input_mem = np.zeros(len(self.input_spikes))
        self._exp_input_syn = np.zeros(len(self.input_spikes))
        for idx, spike in enumerate(self.input_spikes):
            self._exp_input_mem[idx] = np.e**(spike.time/self.parameters.tau_mem)
            self._exp_input_syn[idx] = np.e**(spike.time/self.parameters.tau_syn)

    def _compute_sums_for_tmax(self):
        self.sum0 = np.zeros((len(self.input_spikes)+1, self.parameters.n))
        self.sum1 = np.zeros((len(self.input_spikes)+1, self.parameters.n))
        for idx, spike in enumerate(self.input_spikes):
            self.sum0[idx+1] = self.sum0[idx] + self._exp_input_syn[idx]*self.w_in[spike.source_neuron, :]
            self.sum1[idx+1] = self.sum1[idx] + self._exp_input_mem[idx]*self.w_in[spike.source_neuron, :]

    def _tmax(self, input_spike_idx : int, target_nrn_idx : int) -> float:
        if input_spike_idx == 0:
            return 0
        elif input_spike_idx == len(self.input_spikes):
            t_input = np.inf
        else:
            t_input = self.input_spikes[input_spike_idx].time
        sum_0 = self.sum0[input_spike_idx, target_nrn_idx]
        sum_1 = self.sum1[input_spike_idx, target_nrn_idx]
        largest_t_pre = self.input_spikes[input_spike_idx-1].time
        for post_spike in self._post_spikes_per_neuron[target_nrn_idx]:
            if t_input <= post_spike.time:
                break
            sum_1 += -1*np.e**(post_spike.time/self.parameters.tau_mem)
        tmax = self._tmax_prefactor * (self._tmax_summand + np.log(sum_1/sum_0))
        if largest_t_pre <= tmax <= t_input:
            return tmax
        return None

    def _v(self, t : float, target_nrn_idx : int) -> float:
        times = np.array([s.time for s in self.input_spikes if s.time <= t])
        sources = [s.source_neuron for s in self.input_spikes if s.time <= t]
        kernels = self._k(t-times)
        v = sum([self.w_in[source, target_nrn_idx]*k for source, k in zip(sources, kernels)])
        post_times = np.array([s.time for s in self._post_spikes_per_neuron[target_nrn_idx] if s.time <= t])
        v += np.sum(-self.parameters.v_th*np.exp(-(t-post_times)/self.parameters.tau_mem))
        return v

    def forward(self, input_spikes : List[Spike], code : str = "cpp") -> List[Spike]:
        super().forward(input_spikes)
        self.input_spikes.sort(key=lambda x: x.time)
        if code == "python":
            self.get_spikes_python(input_spikes)
            self.post_spikes.sort(key=lambda x: x.time)
            return self.post_spikes
        elif code == "cpp":
            self._lif_cpp.set_weights(self.w_in)
            self._lif_cpp.set_input_spikes(input_spikes)
            self._lif_cpp.get_spikes()
            self._post_spikes_per_neuron = self._lif_cpp.post_spikes
            self.post_spikes = [x for l in self._post_spikes_per_neuron for x in l]
            self.post_spikes.sort(key=lambda x: x.time)
            self._ran_forward = True
            return self.post_spikes
        raise RuntimeError(f"Code not recognized: {code}.")

    def get_spikes_python(self, input_spikes : List[Spike]) -> List[Spike]:
        self.post_spikes = list()
        self._post_spikes_per_neuron = [list() for _ in range(self.parameters.n)]
        self._compute_exponentiated_times()
        self._compute_sums_for_tmax()
        for target_nrn_idx in range(self.parameters.n):
            self.get_spikes_for_neuron(target_nrn_idx)
        self._ran_forward = True
        return self.post_spikes


    def _bracket_spike(self, t_before, t_after, target_nrn_idx):
        return brentq(lambda t: self._v(t, target_nrn_idx) - 1, t_before, t_after, disp=True)

    def backward(self, output_spikes : List[Spike] = None, code : str = "cpp"):
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        if output_spikes is not None:
            assert(len(output_spikes) == len(self.post_spikes))
            self.post_spikes = output_spikes
        if code == "cpp":
            self._lif_cpp.set_post_spikes(self.post_spikes)
            self._lif_cpp.get_errors()
            self.input_spikes = self._lif_cpp.input_spikes
            self.gradient = self._lif_cpp.gradient
        else:
            for target_nrn_idx in range(self.parameters.n):
                # iterate pre, post spikes in reverse order
                # join lists and sort by time, descending order
                nrn_events = self.input_spikes + self._post_spikes_per_neuron[target_nrn_idx]
                if len(nrn_events) == 0:
                    continue
                nrn_events.sort(key=lambda x: x.time, reverse=True)
                largest_time = nrn_events[0].time
                lambda_v = 0
                previous_t = -np.inf
                lambda_i_spikes = list()
                # function to check if post spike caused by current neuron
                is_pre_spike = lambda x: x.source_layer != id(self)
                is_my_post_spike = lambda x: x.source_layer == id(self) and x.source_neuron == target_nrn_idx
                for event in nrn_events:
                    t_bwd = largest_time - event.time
                    # pre spike -> gradient sample
                    if is_pre_spike(event):
                        lambda_v = np.exp(-(t_bwd-previous_t)/self.parameters.tau_mem)*lambda_v
                        lambda_i = 0
                        for lambda_i_jump, t_post_bwd in lambda_i_spikes:
                            lambda_i += lambda_i_jump*self._k_bwd(t_bwd - t_post_bwd)
                        self.gradient[event.source_neuron, target_nrn_idx] += -self.parameters.tau_syn*lambda_i
                        outbound_signal = self.w_in[event.source_neuron, target_nrn_idx]*(lambda_v - lambda_i)
                        if event.source_layer is not None:
                            event.error += outbound_signal
                    # post spike -> jump and absorb error
                    elif is_my_post_spike(event):
                        i = self._i(event.time, target_nrn_idx)
                        # decay down first
                        lambda_v = np.exp(-(t_bwd-previous_t)/self.parameters.tau_mem)*lambda_v
                        lambda_i_jump = 1/(i-self.parameters.v_th)*(self.parameters.v_th*lambda_v + event.error)
                        lambda_i_spikes.append((lambda_i_jump, t_bwd))
                        # self-jump and error absorption
                        lambda_v = i/(i-self.parameters.v_th) * lambda_v + 1/(i-self.parameters.v_th) * event.error
                    previous_t = t_bwd
        self._ran_backward = True
        super().backward()

    def get_spikes_for_neuron(self, target_nrn_idx : int):
        if self.w_in is None:
            raise RuntimeError("Set weights first!")
        finished = False
        final_spike = Spike(time=np.finfo(np.float64).max, source_neuron=-1)
        processed_up_to = 0
        while not finished:
            for i, spike in enumerate(self.input_spikes+[final_spike]):
                if i < processed_up_to:
                    continue
                x = self._tmax(i, target_nrn_idx)
                if x is not None:
                    # check for post spike
                    vmax = self._v(x, target_nrn_idx)
                    if vmax > self.parameters.v_th+np.finfo(float).eps:
                        t_before = 0
                        # Find earliest pre/post spike before t_pre
                        for cmp_spike in self.input_spikes:
                            if t_before < cmp_spike.time < spike.time:
                                t_before = cmp_spike.time
                            if cmp_spike.time > spike.time:
                                break
                        for post_spike in self._post_spikes_per_neuron[target_nrn_idx]:
                            if t_before < post_spike.time < spike.time:
                                t_before = post_spike.time
                            if post_spike.time > spike.time:
                                break
                        t0 = self._bracket_spike(t_before, x, target_nrn_idx)
                        post_spike = Spike(time=t0, source_neuron=target_nrn_idx, source_layer=id(self))
                        self.post_spikes.append(post_spike)
                        self._post_spikes_per_neuron[target_nrn_idx].append(post_spike)
                        processed_up_to = i
                        break
                if self._v(spike.time, target_nrn_idx) > self.parameters.v_th+np.finfo(float).eps:
                    t_before = self.input_spikes[i-1].time
                    x = spike.time
                    t0 = self._bracket_spike(t_before, x, target_nrn_idx)
                    post_spike = Spike(time=t0, source_neuron=target_nrn_idx, source_layer=id(self))
                    self.post_spikes.append(post_spike)
                    self._post_spikes_per_neuron[target_nrn_idx].append(post_spike)
                    processed_up_to = i
                    break
                if i == len(self.input_spikes):
                    finished = True
        return self.post_spikes

    def get_voltage_trace_for_neuron(self, input_spikes : List[Spike], target_nrn_idx : int, t_max : float = None, dt : float = 1e-4) -> np.array:
        if self.w_in is None:
            raise RuntimeError("Set weights first!")
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        if t_max is None:
            t_max = np.max([spike.time for spike in input_spikes])
        ts = np.arange(0, t_max, step=dt)
        v = np.zeros_like(ts)
        for t_idx, t in enumerate(ts):
            v[t_idx] = self._v(t, target_nrn_idx)
        return v

    def _i(self, t : float, target_nrn_idx : int) -> float:
        i = 0
        for spike in self.input_spikes:
            if spike.time <= t:
                i += self.w_in[spike.source_neuron, target_nrn_idx]*np.exp(-(t-spike.time)/self.parameters.tau_syn)
        return i
