from typing import Tuple

from .layer import Layer
from .training import AbstractTwoLayer
from .lif_layer import LIFLayer, LIFLayerParameters
from .loss_layer import VMaxCrossEntropyLoss, VMaxCrossEntropyLossParameters


class TwoLayerVMax(AbstractTwoLayer):
    def __init__(
        self,
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(),
        **kwargs,
    ):
        super().__init__(
            hidden_layer_class=LIFLayer,
            hidden_parameters=hidden_parameters,
            loss_class=VMaxCrossEntropyLoss,
            loss_parameters=loss_parameters,
            **kwargs,
        )