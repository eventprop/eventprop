import numpy as np
from scipy.optimize import brentq
import logging
from typing import List, NamedTuple

from .layer import Layer
from .eventprop_cpp import (
    compute_spikes_batch_cpp,
    backward_batch_cpp,
    Spikes,
    SpikesVector,
)
from . import eventprop_cpp


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
        self.post_batch = None
        self.gradient = np.zeros_like(self.w_in)

    def forward(self, input_batch: List[Spikes]):
        super().forward(input_batch)
        self.post_batch = compute_spikes_batch_cpp(
            self.w_in,
            self.input_batch,
            self.parameters.v_th,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )
        self._ran_forward_batch = True
        return self.post_batch

    def backward(self):
        backward_batch_cpp(
            self.input_batch,
            self.post_batch,
            self.w_in,
            self.gradient,
            self.parameters.v_th,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )
        self._ran_backward = True
        super().backward()

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
