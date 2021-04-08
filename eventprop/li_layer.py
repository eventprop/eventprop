import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from .eventprop_cpp import (
    MaximaVector,
    SpikesVector,
    compute_maxima_batch_cpp,
    backward_maxima_batch_cpp,
    compute_voltage_trace_cpp,
)

# fmt: off
class LILayerParameters(NamedTuple):
    n               : int   = None
    n_in            : int   = None
    tau_mem         : float = 20e-3 # s
    tau_syn         : float = 5e-3  # s
    v_leak          : float = 0
    plastic_weights : bool  = True
    w_dist          : WeightDistribution = GaussianDistribution()
# fmt: on


class LILayer(Layer):
    def __init__(self, parameters: LILayerParameters, w_in: np.array = None):
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
        self.input_batch = None
        self.gradient = np.zeros_like(self.w_in)

    def forward(self, input_batch: SpikesVector) -> MaximaVector:
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
            np.inf,
            self.parameters.tau_mem,
            self.parameters.tau_syn,
        )

    def zero_grad(self):
        self.gradient[:] = 0
