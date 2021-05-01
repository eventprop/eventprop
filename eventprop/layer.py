from __future__ import annotations
import numpy as np
from typing import Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .eventprop_cpp import Spikes, SpikesVector

# fmt: off
@dataclass
class SpikeDataset:
    spikes         : SpikesVector
    labels         : np.ndarray

    def __post_init__(self):
        assert len(self.spikes) == self.labels.size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.spikes[key], self.labels[key]
        else:
            return SpikeDataset(self.spikes[key], self.labels[key])

    def __len__(self):
        return len(self.spikes)

    def shuffle(self):
        idxs = np.arange(len(self))
        np.random.shuffle(idxs)
        self.spikes = SpikesVector(np.array(self.spikes)[idxs].tolist())
        self.labels = self.labels[idxs]
# fmt: on


@dataclass
class WeightDistribution(ABC):
    seed: int = None

    def __post_init__(self):
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass


@dataclass
class GaussianDistribution(WeightDistribution):
    w_mean: float = 0
    w_std: float = 1

    def get_weights(self, n_in: int, n_out: int) -> np.ndarray:
        return self.rng.normal(self.w_mean, self.w_std, (n_in, n_out))


@dataclass
class UniformDistribution(WeightDistribution):
    w_lower: float = -1
    w_upper: float = 1

    def get_weights(self, n_in: int, n_out: int):
        return self.rng.uniform(self.w_lower, self.w_upper, size=(n_in, n_out))


@dataclass
class DiagonalWeights(WeightDistribution):
    def get_weights(self, n_in: int, n_out: int):
        assert n_in == n_out
        return np.diag(np.ones(n_in))


class Layer(ABC):
    def __init__(self):
        self.input_batch = None
        self.ancestor_layer = None
        self._ran_forward = False
        self._ran_backward = False

    def __call__(
        self,
        arg: Union[SpikesVector, Tuple[SpikesVector, Layer]],
    ) -> Union[Tuple[Spikes, Layer], Tuple[SpikesVector, Layer]]:
        if isinstance(arg, tuple):
            if isinstance(arg[0], SpikesVector) and isinstance(arg[1], Layer):
                self.ancestor_layer = arg[1]
                return self.forward(arg[0]), self
            raise RuntimeError("Arguments not recognized.")
        elif isinstance(arg, SpikesVector):
            return self.forward(arg), self
        raise RuntimeError("Arguments not recognized.")

    @abstractmethod
    def forward(self, input_batch: SpikesVector):
        self.input_batch = input_batch

    @abstractmethod
    def backward(self):
        if self._ran_backward is False:
            raise RuntimeError("Run backward to create errors first.")
        if self.ancestor_layer is not None:
            self.ancestor_layer.backward()
