from __future__ import annotations
import numpy as np
from typing import List, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from eventprop.lif_layer_cpp import Spike

# fmt: off
@dataclass
class SpikePattern:
    spikes         : List[Spike]
    label          : int

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
        self, arg: Union[List[Spike], Tuple[List[Spike], Layer]]
    ) -> Tuple[List[Spike], Layer]:
        if isinstance(arg, tuple):
            if isinstance(arg[0], list) and isinstance(arg[1], Layer):
                self.ancestor_layer = arg[1]
                input_spikes = arg[0]
            else:
                raise RuntimeError("Arguments not recognized.")
        elif isinstance(arg, list):
            input_spikes = arg
        return self.forward(input_spikes), self

    @abstractmethod
    def forward(self, input_spikes: List[Spike]):
        self.input_spikes = input_spikes

    @abstractmethod
    def backward(self):
        if self._ran_backward is False:
            raise RuntimeError("Run backward to create errors first.")
        if self.ancestor_layer is not None:
            self.ancestor_layer.backward(self.input_spikes)
