from typing import Tuple

from .layer import Layer
from .training import AbstractTwoLayer, AbstractOneLayer
from .lif_layer import LIFLayer, LIFLayerParameters
from .loss_layer import VMaxCrossEntropyLoss, VMaxCrossEntropyLossParameters


class OneLayerVMax(AbstractOneLayer):
    def __init__(
        self,
        output_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(),
        **kwargs,
    ):
        super().__init__(
            output_layer_class=LIFLayer,
            output_parameters=output_parameters,
            loss_class=VMaxCrossEntropyLoss,
            loss_parameters=loss_parameters,
            **kwargs,
        )

    def get_weight_copy(self) -> Tuple:
        return (self.output_layer.w_in.copy(), self.loss.w_in.copy())