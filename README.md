# EventProp
A simple event-based simulator that implements the EventProp algorithm for backpropagation in spiking neural networks.

## Installation
Requires Python >= 3.8 and Boost.
```python
pip install -U .
```

## Quickstart
Creates a two-layer network, a simple spike pattern and does a forward/backward pass.
```python
from eventprop.layer import SpikePattern
from eventprop.lif_layer_cpp import Spike
from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.loss_layer import TTFSCrossEntropyLoss, TTFSCrossEntropyLossParameters

hidden_layer = LIFLayer(LIFLayerParameters(n_in=10, n=5, w_mean=5, w_std=1))
output_layer = LIFLayer(LIFLayerParameters(n_in=5, n=3, w_mean=5, w_std=1))
loss = TTFSCrossEntropyLoss(TTFSCrossEntropyLossParameters(n=3))

pattern = SpikePattern(spikes=[Spike(time=0.01*x, source_neuron=x) for x in range(10)], label=1)

loss(output_layer(hidden_layer(pattern.spikes)))
loss.backward(pattern.label)

print(f"First spikes: {loss.first_spikes}")
print(f"Got loss: {loss.get_loss(pattern.label)}, classification result: {loss.get_classification_result(pattern.label)}")

```
This prints:
```
First spikes: [<Spike w/ source_neuron=0, time=0.0107255, source_layer=140226107226048, error=-125.937>, <Spike w/ source_neuron=1, time=0.0102994, source_layer=140226107226048, error=345.556>, <Spike w/ source_neuron=2, time=0.00962609, source_layer=140226107226048, error=-218.219>]
Got loss: 1.1747556275433069, classification result: 0
```
