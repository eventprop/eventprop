# EventProp
A simple event-based simulator that implements the EventProp algorithm for backpropagation in spiking neural networks.

## Installation
Requires Python >= 3.8 and Boost.
```python
pip install -U git+git://github.com/neurognosis/eventprop
```

## Quickstart
### Forward/Backward Pass
Creates a two-layer network, a simple spike pattern and does a forward/backward pass to compute the gradient with respect to a cross-entropy loss function over the "Time-To-First-Spike" (TTFS).
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
The gradients are now available in `hidden_layer.gradient`, `output_layer.gradient`. The output is:
```
First spikes: [<Spike w/ source_neuron=0, time=0.0107255, source_layer=140226107226048, error=-125.937>, <Spike w/ source_neuron=1, time=0.0102994, source_layer=140226107226048, error=345.556>, <Spike w/ source_neuron=2, time=0.00962609, source_layer=140226107226048, error=-218.219>]
Got loss: 1.1747556275433069, classification result: 0
```

### Training of Two-Layer Network
In order to train a two-layer network using a TTFS cross-entropy loss, subclass `TwoLayerTTFS` and provide the `load_data` function that stores lists of `SpikePattern` in `train_spikes`, `test_spikes`, `valid_spikes`.
Training can be started using the `train` function.
An example from `yinyang.py`:
```python
 class YinYangTTFS(TwoLayerTTFS):
    def __init__(self, gd_parameters : GradientDescentParameters = GradientDescentParameters(batch_size=200, iterations=10000, lr=0.01, gradient_clip=None),
    hidden_parameters : LIFLayerParameters = LIFLayerParameters(n_in=5, n=200, w_mean=2.5, w_std=1.5, tau_mem=20e-3, tau_syn=5e-3),
    output_parameters : LIFLayerParameters = LIFLayerParameters(n_in=200, n=3, w_mean=1., w_std=1., tau_mem=20e-3, tau_syn=5e-3),
    loss_parameters : TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(n=3),
    t_min : float = 10e-3,
    t_max : float = 40e-3,
    t_bias : float = 20e-3,
     **kwargs):
        self.t_min, self.t_max, self.t_bias = t_min, t_max, t_bias
        super().__init__(gd_parameters=gd_parameters, hidden_parameters=hidden_parameters, output_parameters=output_parameters, loss_parameters=loss_parameters, **kwargs)

    def load_data(self):
        train_samples = np.load(os.path.join(dir_path, "train_samples.npy"))
        test_samples = np.load(os.path.join(dir_path, "test_samples.npy"))
        valid_samples = np.load(os.path.join(dir_path, "validation_samples.npy"))
        train_labels = np.load(os.path.join(dir_path, "train_labels.npy"))
        test_labels = np.load(os.path.join(dir_path, "test_labels.npy"))
        valid_labels = np.load(os.path.join(dir_path, "validation_labels.npy"))
        def get_patterns(samples, labels):
            patterns = list()
            for s, l in zip(samples, labels):
                spikes = [Spike(time=self.t_min+x*(self.t_max-self.t_min), source_neuron=idx) for idx, x in enumerate(s)]
                spikes += [Spike(time=self.t_bias, source_neuron=len(s))]
                spikes.sort(key=lambda x: x.time)
                patterns.append(SpikePattern(spikes, l))
            return patterns
        self.train_spikes, self.test_spikes, self.valid_spikes = get_patterns(train_samples, train_labels), get_patterns(test_samples, test_labels), get_patterns(valid_samples, valid_labels)
```
