import logging
from time import sleep
from dask.distributed import Client, LocalCluster
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

from eventprop.yinyang import do_single_run, dir_path

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cluster = LocalCluster(n_workers=10, threads_per_worker=1)
    client = Client(cluster)
    seeds = 10
    results = list()
    for seed in range(seeds):
        results.append(client.submit(do_single_run, seed, f"yinyang_{seed}.pkl"))
    while not all([x.done() for x in results]):
        sleep(0.1)

    valid_labels = np.load(join(dir_path, "validation_labels.npy"))
    valid_samples = np.load(join(dir_path, "validation_samples.npy"))

    all_valid_losses = list()
    all_valid_errors = list()
    normalized_times_0 = list()
    normalized_times_1 = list()
    normalized_times_2 = list()
    for idx in range(10):
        _, _, _, _, _, accs, losses, first_times, _ = pickle.load(open(f"yinyang_{idx}.pkl", "rb"))
        errors = 1 - np.array(accs)
        all_valid_errors.append(errors)
        all_valid_losses.append(losses)
        if idx == 0:
            for patterns in first_times[-1]:
                all_times = [x.time for x in patterns if x is not None]
                if len(all_times) == 0:
                    normalized_times_0.append(np.nan)
                    normalized_times_1.append(np.nan)
                    normalized_times_2.append(np.nan)
                    continue
                min_time = min([x.time for x in patterns if x is not None])
                def get_time(x):
                    if x is None:
                        return np.nan
                    else:
                        return x.time - min_time
                normalized_times_0.append(get_time(patterns[0]))
                normalized_times_1.append(get_time(patterns[1]))
                normalized_times_2.append(get_time(patterns[2]))

    def plot_times(times, fname):
        plt.figure(figsize=(4,3))
        for x, y, val in zip(valid_samples[:,0], valid_samples[:,1], times):
            if np.isnan(val):
                plt.scatter(x,y,color="black",marker="x", alpha=0.5)
        plt.scatter(valid_samples[:,0], valid_samples[:,1], c=times, cmap="inferno_r", edgecolors="black", vmin=0,vmax=0.03)
        cbar=plt.colorbar(ticks=[0,0.03], label="$\\Delta t$ [$t_\\mathrm{max}-t_\\mathrm{min}$]")
        cbar.set_ticklabels(["0", "1"])
        plt.xticks([0,1])
        plt.yticks([0,1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(fname)
    plot_times(normalized_times_0, "delta_t_0.pdf")
    plot_times(normalized_times_1, "delta_t_1.pdf")
    plot_times(normalized_times_2, "delta_t_2.pdf")

    plt.figure(figsize=(4,3))
    for errors in all_valid_errors:
        plt.plot(errors, "k-", alpha=0.1)
    plt.plot(np.mean(all_valid_errors, axis=0))
    plt.ylim(0.01, 1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Error")
    plt.savefig("yinyang_errors.pdf")
    print(f"Error statistics: {np.mean(np.array(all_valid_errors)[:, -1])} +- {np.std(np.array(all_valid_errors)[:, -1])}")
    print(f"Accuracy statistics: {np.mean(1-np.array(all_valid_errors)[:, -1])} +- {np.std(1-np.array(all_valid_errors)[:, -1])}")

    plt.figure(figsize=(4,3))
    for losses in all_valid_losses:
        plt.plot(losses, "k-", alpha=0.1)
    plt.plot(np.mean(all_valid_losses, axis=0))
    plt.ylim(0.1, 2)
    plt.yticks([0.1, 1])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.savefig("yinyang_loss.pdf")