from .training import AbstractTwoLayer
from .lif_layer import LIFLayer, LIFLayerParameters
from .loss_layer import TTFSCrossEntropyLoss, TTFSCrossEntropyLossParameters


class TwoLayerTTFS(AbstractTwoLayer):
    def __init__(
        self,
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(),
        output_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(),
        **kwargs,
    ):
        super().__init__(
            hidden_layer_class=LIFLayer,
            output_layer_class=LIFLayer,
            loss_class=TTFSCrossEntropyLoss,
            hidden_parameters=hidden_parameters,
            output_parameters=output_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )