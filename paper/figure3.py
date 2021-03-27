from eventprop.layer import GaussianDistribution
from eventprop.lif_layer import LIFLayerParameters
from eventprop.loss_layer import (
    VMaxCrossEntropyLossParameters,
    TTFSCrossEntropyLossParameters,
)
import logging
from time import sleep
from dask.distributed import Client, LocalCluster
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists

from eventprop.mnist import OneLayerMNISTVMax, TwoLayerMNISTTTFS
from eventprop.training import GradientDescentParameters


def do_single_run_ttfs(seed, save_to):
    np.random.seed(seed)
    mnist = TwoLayerMNISTTTFS(
        gd_parameters=GradientDescentParameters(
            minibatch_size=256,
            iterations=int(60000 / 256) * 100,
            lr=1e-3,
            gradient_clip=None,
        ),
        loss_parameters=TTFSCrossEntropyLossParameters(
            n=10,
            alpha=0.0006,
            tau0=0.0027,
            tau1=0.0086,
        ),
        output_parameters=LIFLayerParameters(
            n=10,
            n_in=350,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(seed, 0.0, 0.036),
        ),
        hidden_parameters=LIFLayerParameters(
            n=350,
            n_in=784,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(seed, 0, 0.71),
        ),
        weight_increase_threshold_output=0.24,
        weight_increase_bump=1e-5,
        lr_decay_gamma=1,
    )
    mnist.train(
        test_every=int(60000 / 256),
        valid_every=None,
        save_to=save_to,
        save_every=1000,
        save_final_weights_only=True,
    )


def do_single_run_vmax(seed, save_to):
    np.random.seed(seed)
    mnist = OneLayerMNISTVMax(
        gd_parameters=GradientDescentParameters(
            minibatch_size=256,
            iterations=int(60000 / 256) * 100,
            lr=1e-3,
            gradient_clip=None,
        ),
        loss_parameters=VMaxCrossEntropyLossParameters(
            n=10,
            n_in=350,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(seed, 0.0048, 0.0024),
        ),
        output_parameters=LIFLayerParameters(
            n=350,
            n_in=784,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(seed, 0.038, 0.34),
        ),
        weight_increase_threshold_output=1.0,
        weight_increase_bump=0,
        lr_decay_gamma=1,
    )
    mnist.train(
        test_every=int(60000 / 256),
        valid_every=None,
        save_to=save_to,
        save_every=1000,
        save_final_weights_only=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cluster = LocalCluster(n_workers=10, threads_per_worker=1, memory_limit="8GB")
    client = Client(cluster)
    seeds = 10
    results = list()
    for seed in range(seeds):
        fname = f"mnist_ttfs_{seed}.pkl"
        if not exists(fname):
            results.append(client.submit(do_single_run_ttfs, seed, fname))
    while not all([x.done() for x in results]):
        sleep(0.1)
    results = list()
    for seed in range(seeds):
        fname = f"mnist_vmax_{seed}.pkl"
        if not exists(fname):
            results.append(client.submit(do_single_run_vmax, seed, fname))
    while not all([x.done() for x in results]):
        sleep(0.1)

    # valid_labels = np.load(join(dir_path, "validation_labels.npy"))
    # valid_samples = np.load(join(dir_path, "validation_samples.npy"))

    # all_valid_losses = list()
    # all_valid_errors = list()
    # normalized_times_0 = list()
    # normalized_times_1 = list()
    # normalized_times_2 = list()
    # for idx in range(10):
    #    _, _, _, _, _, accs, losses, first_times, _ = pickle.load(
    #        open(f"yinyang_{idx}.pkl", "rb")
    #    )
    #    errors = 1 - np.array(accs)
    #    all_valid_errors.append(errors)
    #    all_valid_losses.append(losses)
    #    if idx == 0:
    #        for patterns in first_times[-1]:
    #            all_times = [x.time for x in patterns if x is not None]
    #            if len(all_times) == 0:
    #                normalized_times_0.append(np.nan)
    #                normalized_times_1.append(np.nan)
    #                normalized_times_2.append(np.nan)
    #                continue
    #            min_time = min([x.time for x in patterns if x is not None])

    #            def get_time(x):
    #                if x is None:
    #                    return np.nan
    #                else:
    #                    return x.time - min_time

    #            normalized_times_0.append(get_time(patterns[0]))
    #            normalized_times_1.append(get_time(patterns[1]))
    #            normalized_times_2.append(get_time(patterns[2]))

    # def plot_times(times, fname):
    #    plt.figure(figsize=(4, 3))
    #    for x, y, val in zip(valid_samples[:, 0], valid_samples[:, 1], times):
    #        if np.isnan(val):
    #            plt.scatter(x, y, color="black", marker="x", alpha=0.5)
    #    plt.scatter(
    #        valid_samples[:, 0],
    #        valid_samples[:, 1],
    #        c=times,
    #        cmap="inferno_r",
    #        edgecolors="black",
    #        vmin=0,
    #        vmax=0.03,
    #    )
    #    cbar = plt.colorbar(
    #        ticks=[0, 0.03], label="$\\Delta t$ [$t_\\mathrm{max}-t_\\mathrm{min}$]"
    #    )
    #    cbar.set_ticklabels(["0", "1"])
    #    plt.xticks([0, 1])
    #    plt.yticks([0, 1])
    #    plt.xlabel("x")
    #    plt.ylabel("y")
    #    plt.savefig(fname)

    # plot_times(normalized_times_0, "delta_t_0.pdf")
    # plot_times(normalized_times_1, "delta_t_1.pdf")
    # plot_times(normalized_times_2, "delta_t_2.pdf")

    # plt.figure(figsize=(4, 3))
    # for errors in all_valid_errors:
    #    plt.plot(errors, "k-", alpha=0.1)
    # plt.plot(np.mean(all_valid_errors, axis=0))
    # plt.ylim(0.01, 1)
    # plt.yscale("log")
    # plt.xlabel("Epoch")
    # plt.ylabel("Validation Error")
    # plt.savefig("yinyang_errors.pdf")
    # print(
    #    f"Error statistics: {np.mean(np.array(all_valid_errors)[:, -1])} +- {np.std(np.array(all_valid_errors)[:, -1])}"
    # )
    # print(
    #    f"Accuracy statistics: {np.mean(1-np.array(all_valid_errors)[:, -1])} +- {np.std(1-np.array(all_valid_errors)[:, -1])}"
    # )

    # plt.figure(figsize=(4, 3))
    # for losses in all_valid_losses:
    #    plt.plot(losses, "k-", alpha=0.1)
    # plt.plot(np.mean(all_valid_losses, axis=0))
    # plt.ylim(0.1, 2)
    # plt.yticks([0.1, 1])
    # plt.yscale("log")
    # plt.xlabel("Epoch")
    # plt.ylabel("Validation Loss")
    # plt.savefig("yinyang_loss.pdf")
