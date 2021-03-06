import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple
import logging

from .layer import Layer
from .lif_layer import LIFLayer


class Optimizer(ABC):
    def __init__(self, loss: Layer):
        self.loss = loss

    def zero_grad(self):
        ancestor_layer = self.loss.ancestor_layer
        while ancestor_layer is not None:
            if not isinstance(ancestor_layer, LIFLayer):
                continue
            ancestor_layer.gradient = np.zeros_like(ancestor_layer.gradient)
            ancestor_layer._lif_cpp.zero_grad()
            ancestor_layer = ancestor_layer.ancestor_layer

    @abstractmethod
    def step(self):
        pass


# fmt: off
class GradientDescentParameters(NamedTuple):
    lr                 : float = 1e-4    # for adam/gd
    minibatch_size     : int   = 100
    iterations         : int   = 1000
    gradient_clip      : float = None
    beta1              : float = 0.9     # for adam
    beta2              : float = 0.999   # for adam
    epsilon            : float = 1e-8    # for adam
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
        ancestor_layer = self.loss.ancestor_layer
        while ancestor_layer is not None:
            if not isinstance(ancestor_layer, LIFLayer):
                continue
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
        ancestor_layer = self.loss.ancestor_layer
        while ancestor_layer is not None:
            if not isinstance(ancestor_layer, LIFLayer):
                continue
            if not hasattr(ancestor_layer, "_opt_adam_m"):
                ancestor_layer._opt_adam_m = np.zeros_like(ancestor_layer.gradient)
            if not hasattr(ancestor_layer, "_opt_adam_v"):
                ancestor_layer._opt_adam_v = np.zeros_like(ancestor_layer.gradient)
            ancestor_layer._opt_adam_m = (
                self.parameters.beta1 * ancestor_layer._opt_adam_m
                + (1 - self.parameters.beta1) * ancestor_layer.gradient
            )
            ancestor_layer._opt_adam_v = (
                self.parameters.beta2 * ancestor_layer._opt_adam_v
                + (1 - self.parameters.beta2) * ancestor_layer.gradient ** 2
            )
            m_hat = ancestor_layer._opt_adam_m / (1 - self.parameters.beta1)
            v_hat = ancestor_layer._opt_adam_v / (1 - self.parameters.beta2)
            delta_w = m_hat / (np.sqrt(v_hat) + self.parameters.epsilon)
            norm_delta_w = np.linalg.norm(delta_w)
            if self.parameters.gradient_clip is not None:
                if norm_delta_w > self.parameters.gradient_clip:
                    delta_w = delta_w / norm_delta_w * self.parameters.gradient_clip
            logging.debug(
                f"Updating weights with delta_w norm {np.linalg.norm(delta_w)}."
            )
            ancestor_layer.w_in += -self.parameters.lr * delta_w
            ancestor_layer = ancestor_layer.ancestor_layer
