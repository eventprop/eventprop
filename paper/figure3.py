from eventprop.layer import GaussianDistribution
from eventprop.lif_layer import LIFLayerParameters
from eventprop.loss_layer import (
    VMaxCrossEntropyLossParameters,
)
import logging
from time import sleep
from dask.distributed import Client, LocalCluster
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from os.path import join, exists

from eventprop.mnist import OneLayerMNISTVMax
from eventprop.training import GradientDescentParameters


def do_single_run_vmax(seed, save_to):
    np.random.seed(seed)
    mnist = OneLayerMNISTVMax(
        gd_parameters=GradientDescentParameters(
            minibatch_size=128,
            epochs=100,
            lr=5e-3,
            gradient_clip=None,
            sigma_jitter=1e-3,
        ),
        loss_parameters=VMaxCrossEntropyLossParameters(
            n=10,
            n_in=350,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(seed, 0.2, 0.37),
        ),
        output_parameters=LIFLayerParameters(
            n=350,
            n_in=784,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(seed, 0.065 * 1.0, 0.045 * 1.0),
        ),
        weight_increase_threshold_output=1.0,
        weight_increase_bump=6.9e-5,
        lr_decay_gamma=0.97,
        lr_decay_step=1,
    )
    mnist.train(
        valid_results_every_epoch=True,
        test_results_every_epoch=False,
        train_results_every_epoch=False,
        save_to=save_to,
        save_every=100,
        save_final_weights_only=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cluster = LocalCluster(n_workers=10, threads_per_worker=1, memory_limit="512GB")
    client = Client(cluster)
    seeds = 10
    results = list()
    for seed in range(seeds):
        fname = f"mnist_vmax_{seed}.pkl"
        if not exists(fname):
            results.append(client.submit(do_single_run_vmax, seed, fname))
        # do_single_run_vmax(seed, "bla.pkl")
    while not all([x.done() for x in results]):
        sleep(0.1)

    all_test_losses = list()
    all_test_errors = list()
    for idx in range(10):
        _, _, accs, losses, _, _, weights = pickle.load(
            open(f"mnist_vmax_{idx}.pkl", "rb")
        )
        errors = 1 - np.array(accs)
        all_test_errors.append(errors)
        all_test_losses.append(losses)
        if idx == 0:
            mnist = OneLayerMNISTVMax()
            mnist.output_layer.w_in = weights[0][0]
            mnist.loss.w_in = weights[0][1]
            mnist.forward(mnist.test_batch)
            predictions = mnist.loss.get_predictions()
            confusion_matrix = np.zeros((10, 10))
            for pred, label in zip(predictions, mnist.test_batch.labels):
                confusion_matrix[pred, label] += 1
            confusion_matrix /= np.sum(confusion_matrix, axis=1)[None, :]
            print(confusion_matrix)
            plt.figure(figsize=(4, 3))
            sn.heatmap(confusion_matrix, annot=True, square=True, annot_kws={"size": 4})
            # plt.colorbar()
            plt.gca().set_aspect("equal")
            plt.savefig("mnist_confusion.pdf")
            for sample_idx in range(2):
                plt.figure(figsize=(4, 3))
                spikes, label = mnist.test_batch[sample_idx]
                for class_idx in range(10):
                    if label == class_idx:
                        ls = "-"
                        alpha = 1
                    else:
                        ls = "--"
                        alpha = 0.5
                    ts, vs = mnist.loss.get_voltage_trace_for_neuron(
                        sample_idx, class_idx, 0.05, dt=1e-5
                    )
                    lines = plt.plot(ts, vs, label=f"{class_idx}")
                    # plt.gca().annotate(
                    #    f"{class_idx}",
                    #    xy=(
                    #        1,
                    #        lines[0].get_ydata()[
                    #            np.argmax(np.abs(lines[0].get_ydata()))
                    #        ],
                    #    ),
                    #    xytext=(6, 0),
                    #    color=lines[0].get_color(),
                    #    xycoords=plt.gca().get_yaxis_transform(),
                    #    textcoords="offset points",
                    #    size=10,
                    #    va="center",
                    # )
                plt.yticks([])
                plt.xlabel("$t$ [s]")
                plt.ylabel("V")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"mnist_v_{sample_idx}.pdf")
                plt.figure(figsize=(4, 4))
                sample = np.full(784, np.nan)
                sample[spikes.sources] = spikes.times
                plt.imshow(sample.reshape((28, 28)))
                plt.gca().set_aspect("equal")
                plt.savefig(f"mnist_sample_{sample_idx}.pdf")

    plt.figure(figsize=(4, 3))
    for errors in all_test_errors:
        plt.plot(errors, "k-", alpha=0.1)
    plt.plot(np.mean(all_test_errors, axis=0))
    min_idx = np.argmin(np.mean(all_test_errors, axis=0))
    min_err = np.mean(all_test_errors, axis=0)[min_idx]
    print(f"Maximum accuracy in epoch {min_idx}: {1-min_err}")
    plt.ylim(0.01, 1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Test Error")
    plt.savefig("mnist_errors.pdf")

    plt.figure(figsize=(4, 3))
    for losses in all_test_losses:
        plt.plot(losses, "k-", alpha=0.1)
    plt.plot(np.mean(all_test_losses, axis=0))
    plt.ylim(0.1, 2)
    plt.yticks([0.1, 1])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.savefig("mnist_loss.pdf")
    print(
        f"Error statistics: {np.mean(np.array(all_test_errors)[:, -1])} +- {np.std(np.array(all_test_errors)[:, -1])}"
    )
    print(
        f"Accuracy statistics: {np.mean(1-np.array(all_test_errors)[:, -1])*100:.2f} +- {np.std(1-np.array(all_test_errors)[:, -1])*100:.2f}"
    )
