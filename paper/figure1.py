import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.eventprop_cpp import Spikes, SpikesVector


def get_poisson_times(isi, t_max):
    times = [np.random.exponential(isi)]
    while times[-1] < t_max:
        times.append(times[-1] + np.random.exponential(isi))
    return times[:-1]


def get_poisson_spikes(isi, t_max, n):
    all_times = np.array([])
    all_sources = np.array([])
    for nrn_idx in range(n):
        times = get_poisson_times(isi, t_max)
        all_times = np.concatenate([all_times, times])
        all_sources = np.concatenate([all_sources, np.full(len(times), nrn_idx)])
    sort_idxs = np.argsort(all_times)
    all_times = all_times[sort_idxs]
    all_sources = all_sources[sort_idxs]
    return SpikesVector(
        [
            Spikes(
                all_times.astype(np.float64),
                all_sources.astype(np.int32),
            )
        ]
    )


if __name__ == "__main__":
    np.random.seed(6)

    hidden_parameters = LIFLayerParameters(n_in=100, n=1)
    output_parameters = LIFLayerParameters(n_in=1, n=1)
    w_hidden = np.random.normal(
        0.005, 0.05, size=(hidden_parameters.n_in, hidden_parameters.n)
    )
    w_output = np.random.normal(
        5.0, 0.0, size=(output_parameters.n_in, output_parameters.n)
    )

    w_eps = 1e-6
    isi = 5e-3
    t_max = 0.5
    plot_t_max = 0.6

    input_spikes = get_poisson_spikes(isi, t_max, hidden_parameters.n_in)

    # Calculate numerical gradient for hidden synapse
    w_plus = w_hidden.copy()
    w_plus[0, 0] += w_eps
    hidden_layer = LIFLayer(hidden_parameters, w_plus)
    output_layer = LIFLayer(output_parameters, w_output)
    output_layer(hidden_layer(input_spikes))
    for spike_idx in range(output_layer.post_batch[0].n_spikes):
        output_layer.post_batch[0].set_error(spike_idx, 1)
    output_layer.backward()
    t_plus = np.sum(output_layer.post_batch[0].times)

    w_minus = w_hidden.copy()
    w_minus[0, 0] -= w_eps
    hidden_layer = LIFLayer(hidden_parameters, w_minus)
    output_layer = LIFLayer(output_parameters, w_output)
    output_layer(hidden_layer(input_spikes))
    for spike_idx in range(output_layer.post_batch[0].n_spikes):
        output_layer.post_batch[0].set_error(spike_idx, 1)
    output_layer.backward()
    t_minus = np.sum(output_layer.post_batch[0].times)
    numerical_hidden_grad = (t_plus - t_minus) / (2 * w_eps)

    # Calculate numerical gradient for output synapse
    w_plus = w_output + w_eps
    hidden_layer = LIFLayer(hidden_parameters, w_hidden)
    output_layer = LIFLayer(output_parameters, w_plus)
    output_layer(hidden_layer(input_spikes))
    for spike_idx in range(output_layer.post_batch[0].n_spikes):
        output_layer.post_batch[0].set_error(spike_idx, 1)
    output_layer.backward()
    t_plus = np.sum(output_layer.post_batch[0].times)

    w_minus = w_output - w_eps
    hidden_layer = LIFLayer(hidden_parameters, w_hidden)
    output_layer = LIFLayer(output_parameters, w_minus)
    output_layer(hidden_layer(input_spikes))
    for spike_idx in range(output_layer.post_batch[0].n_spikes):
        output_layer.post_batch[0].set_error(spike_idx, 1)
    output_layer.backward()
    t_minus = np.sum(output_layer.post_batch[0].times)
    numerical_output_grad = (t_plus - t_minus) / (2 * w_eps)

    # Get adjoint gradient and variables
    hidden_layer = LIFLayer(hidden_parameters, w_hidden)
    output_layer = LIFLayer(output_parameters, w_output)

    output_layer(hidden_layer(input_spikes))
    output_spikes = output_layer.post_batch[0]
    hidden_spikes = hidden_layer.post_batch[0]
    for spike_idx in range(output_spikes.n_spikes):
        output_layer.post_batch[0].set_error(spike_idx, 1)
    output_layer.backward()
    output_gradient = output_layer.gradient.copy()
    hidden_gradient = hidden_layer.gradient.copy()
    ts, output_voltage_trace = output_layer.get_voltage_trace_for_neuron(
        0, 0, plot_t_max, dt=1e-4
    )
    ts, hidden_voltage_trace = hidden_layer.get_voltage_trace_for_neuron(
        0, 0, plot_t_max, dt=1e-4
    )

    ts, output_lambda_trace = output_layer.get_lambda_i_trace_for_neuron(
        0, 0, plot_t_max, dt=1e-4
    )
    output_lambdas = [
        output_layer.get_lambda_i_for_neuron(0, 0, t)
        for nrn, t in zip(hidden_spikes.sources, hidden_spikes.times)
        if nrn == 0
    ]
    ts, hidden_lambda_trace = hidden_layer.get_lambda_i_trace_for_neuron(
        0, 0, plot_t_max, dt=1e-4
    )
    hidden_lambdas = [
        hidden_layer.get_lambda_i_for_neuron(0, 0, t)
        for nrn, t in zip(input_spikes[0].sources, input_spikes[0].times)
        if nrn == 0
    ]

    def plot(
        ts,
        pre_spike_times,
        voltage_trace,
        lambda_trace,
        lambdas,
        tau_syn,
        gradient,
        numerical_gradient,
        save_to,
    ):
        _, axs = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(10, 3))
        axs = axs.flat
        plt.sca(axs[0])
        plt.plot(ts, voltage_trace)
        plt.ylabel("V")
        plt.xlabel("Time [s]")
        plt.sca(axs[1])
        plt.plot(ts, lambda_trace)
        plt.ylabel("$\\lambda_I$")
        plt.xlabel("Time [s]")
        plt.sca(axs[2])
        grad = 0
        old_t = ts[-1]
        plt.axhline(
            numerical_gradient,
            linestyle="--",
            label="Finite Differences",
            zorder=0,
            color="black",
        )
        for lambda_i, t in reversed(list(zip(lambdas, pre_spike_times))):
            plt.hlines(grad, t, old_t, zorder=1)
            grad += -tau_syn * lambda_i
            old_t = t
        np.testing.assert_almost_equal(grad, gradient[0])
        plt.hlines(grad, 0, pre_spike_times[0], zorder=1)
        plt.ylabel("Accumulated Gradient")
        plt.xlabel("Time [s]")
        plt.legend(loc="center right")
        plt.tight_layout()
        plt.savefig(save_to)

    print(
        f"Relative deviation of adjoint gradient from numerical: {abs(output_gradient[0,0]-numerical_output_grad)/numerical_output_grad} (output weight)"
    )
    print(
        f"Relative deviation of adjoint gradient from numerical: {abs(hidden_gradient[0,0]-numerical_hidden_grad)/numerical_hidden_grad} (hidden weight)"
    )

    plot(
        ts,
        hidden_spikes.times,
        output_voltage_trace,
        output_lambda_trace,
        output_lambdas,
        output_layer.parameters.tau_syn,
        output_gradient,
        numerical_output_grad,
        "output_trace.svg",
    )
    plot(
        ts,
        input_spikes[0].times[input_spikes[0].sources == 0],
        hidden_voltage_trace,
        hidden_lambda_trace,
        hidden_lambdas,
        hidden_layer.parameters.tau_syn,
        hidden_gradient,
        numerical_hidden_grad,
        "hidden_trace.svg",
    )
