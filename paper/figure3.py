import logging
from time import sleep
from dask.distributed import Client, LocalCluster
import pickle
import numpy as np
import matplotlib.pyplot as plt

from eventprop.yinyang import do_single_run

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

    all_valid_losses = list()
    all_valid_errors = list()
    for idx in range(10):
        _, _, _, _, accs, losses, _ = pickle.load(open(f"yinyang_{idx}.pkl", "rb"))
        errors = 1 - np.array(accs)
        all_valid_errors.append(errors)
        all_valid_losses.append(losses)
    plt.figure(figsize=(4, 3))
    for errors in all_valid_errors:
        plt.plot(errors, "k-", alpha=0.1)
    plt.plot(np.mean(all_valid_errors, axis=0))
    plt.ylim(0.01, 1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Error")
    plt.savefig("yinyang_errors.pdf")
    print(
        f"Error statistics: {np.mean(np.array(all_valid_errors)[:, -1])} +- {np.std(np.array(all_valid_errors)[:, -1])}"
    )

    plt.figure(figsize=(4, 3))
    for losses in all_valid_losses:
        plt.plot(losses, "k-", alpha=0.1)
    plt.plot(np.mean(all_valid_losses, axis=0))
    plt.ylim(0.1, 2)
    plt.yticks([0.1, 1])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.savefig("yinyang_loss.pdf")
