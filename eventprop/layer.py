from __future__ import annotations
import numpy as np
from typing import List, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

# fmt: off
@dataclass
class Spikes:
    times          : np.ndarray
    sources        : np.ndarray
    errors         : np.ndarray = None
    source_layer   : int        = 0
    label          : int        = None
    n_spikes       : int        = None

    def __post_init__(self):
        assert len(self.times) == len(self.sources)
        self.sources = self.sources.astype(np.int32, copy=False)
        self.times = self.times.astype(np.float64, copy=False)
        if self.n_spikes is None:
            self.n_spikes = self.times.size
        if self.errors is None:
            self.errors = np.zeros(self.n_spikes, dtype=np.float64)
        elif self.errors is not None:
            self.errors = self.errors.astype(np.float64, copy=False)
@dataclass
class VMax:
    time           : float
    value          : float
    error          : float
# fmt: on


class Layer(ABC):
    def __init__(self):
        self.input_spikes = None
        self.ancestor_layer = None
        self._ran_forward = False
        self._ran_backward = False

    def __call__(
        self, arg: Union[Spikes, Tuple[Spikes, Layer]]
    ) -> Tuple[Spikes, Layer]:
        if isinstance(arg, tuple):
            if isinstance(arg[0], Spikes) and isinstance(arg[1], Layer):
                self.ancestor_layer = arg[1]
                input_spikes = arg[0]
            else:
                raise RuntimeError("Arguments not recognized.")
        elif isinstance(arg, Spikes):
            input_spikes = arg
        return self.forward(input_spikes), self

    @abstractmethod
    def forward(self, input_spikes: Spikes):
        self.input_spikes = input_spikes

    @abstractmethod
    def backward(self):
        if self._ran_backward is False:
            raise RuntimeError("Run backward to create errors first.")
        if self.ancestor_layer is not None:
            self.ancestor_layer.backward()
