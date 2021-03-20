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


class TwoLayerVMax(AbstractTwoLayer):
    def __init__(
        self,
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(),
        output_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(),
        **kwargs,
    ):
        super().__init__(
            hidden_layer_class=LIFLayer,
            output_layer_class=LIFLayer,
            loss_class=VMaxCrossEntropyLoss,
            hidden_parameters=hidden_parameters,
            output_parameters=output_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )
