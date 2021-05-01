import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple

from .layer import Layer


class Optimizer(ABC):
    def __init__(self, loss: Layer):
        self.loss = loss

    def zero_grad(self):
        ancestor_layer = self.loss
        while ancestor_layer is not None:
            if hasattr(ancestor_layer, "gradient"):
                ancestor_layer.gradient[:] = 0
            ancestor_layer = ancestor_layer.ancestor_layer

    @abstractmethod
    def step(self):
        pass


# fmt: off
class GradientDescentParameters(NamedTuple):
    lr                 : float = 1e-4    # for adam/gd
    minibatch_size     : int   = 100
    epochs             : int   = 100
    beta1              : float = 0.9     # for adam
    beta2              : float = 0.999   # for adam
    epsilon            : float = 1e-8    # for adam
    input_dropout      : float = 0
# fmt: on


class GradientDescent(Optimizer):
    def __init__(
        self,
        loss: Layer,
        parameters: GradientDescentParameters = GradientDescentParameters(),
    ):
        super().__init__(loss)
        self.parameters = parameters

    def step(self):
        ancestor_layer = self.loss
        while ancestor_layer is not None:
            if hasattr(ancestor_layer, "gradient"):
                ancestor_layer.w_in += -self.parameters.lr * ancestor_layer.gradient
            ancestor_layer = ancestor_layer.ancestor_layer


class Adam(Optimizer):
    def __init__(
        self,
        loss: Layer,
        parameters: GradientDescentParameters = GradientDescentParameters(),
    ):
        super().__init__(loss)
        self.parameters = parameters

    def step(self):
        ancestor_layer = self.loss
        while ancestor_layer is not None:
            if not hasattr(ancestor_layer, "gradient"):
                ancestor_layer = ancestor_layer.ancestor_layer
                continue
            if not ancestor_layer.parameters.plastic_weights:
                ancestor_layer = ancestor_layer.ancestor_layer
                continue
            if not hasattr(ancestor_layer, "_opt_adam_m"):
                ancestor_layer._opt_adam_m = np.zeros_like(ancestor_layer.gradient)
            if not hasattr(ancestor_layer, "_opt_adam_v"):
                ancestor_layer._opt_adam_v = np.zeros_like(ancestor_layer.gradient)
            if not hasattr(ancestor_layer, "_opt_adam_i"):
                ancestor_layer._opt_adam_i = 0
            ancestor_layer._opt_adam_i += 1
            ancestor_layer._opt_adam_m = (
                self.parameters.beta1 * ancestor_layer._opt_adam_m
                + (1 - self.parameters.beta1) * ancestor_layer.gradient
            )
            ancestor_layer._opt_adam_v = (
                self.parameters.beta2 * ancestor_layer._opt_adam_v
                + (1 - self.parameters.beta2) * ancestor_layer.gradient ** 2
            )
            m_hat = ancestor_layer._opt_adam_m / (
                1 - self.parameters.beta1 ** ancestor_layer._opt_adam_i
            )
            v_hat = ancestor_layer._opt_adam_v / (
                1 - self.parameters.beta2 ** ancestor_layer._opt_adam_i
            )
            delta_w = m_hat / (np.sqrt(v_hat) + self.parameters.epsilon)

            ancestor_layer.w_in += -self.parameters.lr * delta_w
            ancestor_layer = ancestor_layer.ancestor_layer
