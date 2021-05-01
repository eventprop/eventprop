import numpy as np
from scipy.optimize import brentq
import logging
from typing import List, NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from .eventprop_cpp import (
    compute_spikes_batch_cpp,
    backward_spikes_batch_cpp,
    SpikesVector,
    compute_voltage_trace_cpp,
    compute_lambda_i_cpp,
    compute_lambda_i_trace_cpp,
)
from . import eventprop_cpp


# fmt: off
class LIFLayerParameters(NamedTuple):
    n               : int   = None
    n_in            : int   = None
    tau_mem         : float = 20e-3 # s
    tau_syn         : float = 5e-3  # s
    v_th            : float = 1.
    v_leak          : float = 0
    plastic_weights : bool  = True
    w_dist          : WeightDistribution = GaussianDistribution()
# fmt: on


class LIFLayer(Layer):
    def __init__(self, parameters: LIFLayerParameters, w_in: np.array = None):
        super().__init__()
        self.parameters = parameters
        if w_in is not None:
            assert isinstance(w_in, np.ndarray)
            assert w_in.shape == (self.parameters.n_in, self.parameters.n)
            self.w_in = w_in
        else:
            self.w_in = self.parameters.w_dist.get_weights(
                self.parameters.n_in, self.parameters.n
            )
        self.post_batch = None
        self.gradient = np.zeros_like(self.w_in)

    def forward(self, input_batch: SpikesVector) -> SpikesVector:
        super().forward(input_batch)
        self.post_batch = compute_spikes_batch_cpp(
            self.w_in,
            self.input_batch,
            self.parameters.v_th,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )
        self._ran_forward = True
        return self.post_batch

    def backward(self):
        backward_spikes_batch_cpp(
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
        batch_idx: int,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")
        return compute_voltage_trace_cpp(
            t_max,
            dt,
            target_nrn_idx,
            self.w_in,
            self.input_batch[batch_idx],
            self.parameters.v_th,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )

    def zero_grad(self):
        self.gradient[:] = 0

    def get_lambda_i_trace_for_neuron(
        self,
        batch_idx: int,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return compute_lambda_i_trace_cpp(
            t_max,
            dt,
            target_nrn_idx,
            self.post_batch[batch_idx],
            self.parameters.v_th,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )

    def get_lambda_i_for_neuron(
        self, batch_idx: int, target_nrn_idx: int, t: float
    ) -> float:
        return compute_lambda_i_cpp(
            t,
            target_nrn_idx,
            self.post_batch[batch_idx],
            self.parameters.v_th,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )

    def _i(self, t: float, target_nrn_idx: int) -> float:
        idxs = np.argwhere(self.input_spikes.times <= t)
        if len(idxs) == 0:
            return 0
        else:
            t_pre_idx = idxs[-1, 0] + 1
        i = np.exp(-t / self.parameters.tau_syn) * self.sum0[t_pre_idx, target_nrn_idx]
        return i
