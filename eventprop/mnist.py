import gzip
import pickle
import urllib.request
import os
from tqdm import tqdm
import numpy as np
import logging
import struct

from eventprop.vmax_training import (
    OneLayerVMax,
    VMaxCrossEntropyLossParameters,
)
from eventprop.loss_layer import TTFSCrossEntropyLossParameters
from eventprop.ttfs_training import TwoLayerTTFS
from eventprop.optimizer import GradientDescentParameters
from eventprop.lif_layer import LIFLayerParameters
from eventprop.eventprop_cpp import Spikes, SpikesVector
from eventprop.layer import GaussianDistribution, SpikeDataset, UniformDistribution


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mnist_dataset")

train_path_images_raw = os.path.join(dir_path, "train-images-idx3-ubyte.gz")
train_path_labels_raw = os.path.join(dir_path, "train-labels-idx1-ubyte.gz")
test_path_images_raw = os.path.join(dir_path, "t10k-images-idx3-ubyte.gz")
test_path_labels_raw = os.path.join(dir_path, "t10k-labels-idx1-ubyte.gz")
train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
all_paths = [
    train_path_images_raw,
    train_path_labels_raw,
    test_path_images_raw,
    test_path_labels_raw,
]
all_urls = [train_images_url, train_labels_url, test_images_url, test_labels_url]

train_path = os.path.join(dir_path, "mnist_train.pkl")
test_path = os.path.join(dir_path, "mnist_test.pkl")
valid_path = os.path.join(dir_path, "mnist_valid.pkl")


class MNISTMixin:
    valid_num: int = 5000
    t_min = 0
    t_max = 30e-3

    def load_data(self):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self._convert_data()
        self.train_batch = pickle.load(open(train_path, "rb"))
        self.test_batch = pickle.load(open(test_path, "rb"))
        self.valid_batch = pickle.load(open(valid_path, "rb"))

    def _convert_data(self):
        def get_file(path, url):
            if not os.path.exists(path):
                logging.info("Downloading dataset...")
                urllib.request.urlretrieve(url, path)

        for path, url in zip(all_paths, all_urls):
            get_file(path, url)

        def process_file(image_file, label_file, n):
            image = gzip.open(image_file)
            image.read(16)
            label = gzip.open(label_file)
            label.read(8)
            all_images = list()
            all_labels = list()
            for _ in tqdm(range(n)):
                l = ord(label.read(1))
                im = np.empty(784)
                for idx in range(784):
                    im[idx] = ord(image.read(1))
                all_images.append(im)
                all_labels.append(l)
            return np.array(all_images), np.array(all_labels)

        logging.debug("Loading train set...")
        train_images, train_labels = process_file(
            train_path_images_raw, train_path_labels_raw, 60000
        )
        logging.debug("Loading test set...")
        test_images, test_labels = process_file(
            test_path_images_raw, test_path_labels_raw, 10000
        )
        test_shuffle_idxs = np.arange(10000)
        tmp_rng = np.random.default_rng(0)
        tmp_rng.shuffle(test_shuffle_idxs)
        test_images = test_images[test_shuffle_idxs]
        test_labels = test_labels[test_shuffle_idxs]

        def get_batch(samples, labels):
            spikes = list()
            for s in tqdm(samples):
                times, sources = list(), list()
                for idx, pix in enumerate(s):
                    if pix > 1:
                        times.append(
                            self.t_min + (1 - pix / 255) * (self.t_max - self.t_min)
                        )
                        sources.append(idx)
                times = np.array(times, dtype=np.float64)
                sources = np.array(sources, dtype=np.int32)
                sort_idxs = np.argsort(times)
                times = times[sort_idxs]
                sources = sources[sort_idxs]
                spikes.append(Spikes(times, sources))
            return SpikeDataset(SpikesVector(spikes), labels)

        logging.debug("Creating train set spikes...")
        train_batch = get_batch(train_images, train_labels)
        logging.debug("Creating valid set spikes...")
        valid_batch = get_batch(
            test_images[: self.valid_num], test_labels[: self.valid_num]
        )
        logging.debug("Creating test set spikes...")
        test_batch = get_batch(
            test_images[self.valid_num :], test_labels[self.valid_num :]
        )
        with open(train_path, "wb") as f:
            pickle.dump(train_batch, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_batch, f)
        with open(valid_path, "wb") as f:
            pickle.dump(valid_batch, f)


class TwoLayerMNISTTTFS(MNISTMixin, TwoLayerTTFS):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(
            minibatch_size=256, epochs=100, lr=1e-3, gradient_clip=None
        ),
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=784,
            n=100,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(
                w_mean=4 * 1 / np.sqrt(700), w_std=2 * 1 / np.sqrt(700)
            ),
        ),
        output_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=100,
            n=10,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(
                w_mean=1 / np.sqrt(100), w_std=1 / np.sqrt(100)
            ),
        ),
        loss_parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(
            n=10,
        ),
        **kwargs,
    ):
        super().__init__(
            gd_parameters=gd_parameters,
            hidden_parameters=hidden_parameters,
            output_parameters=output_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )


class OneLayerMNISTVMax(MNISTMixin, OneLayerVMax):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(
            minibatch_size=256, epochs=100, lr=1e-3, gradient_clip=None
        ),
        output_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=784,
            n=100,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(
                w_mean=4 * 1 / np.sqrt(700), w_std=2 * 1 / np.sqrt(700)
            ),
        ),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(
            n=10,
            n_in=100,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(
                w_mean=0.25 * 1 / np.sqrt(128), w_std=0.25 * 1 / np.sqrt(128)
            ),
        ),
        **kwargs,
    ):
        super().__init__(
            gd_parameters=gd_parameters,
            output_parameters=output_parameters,
            loss_parameters=loss_parameters,
            **kwargs,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    np.random.seed(0)
    mnist = TwoLayerMNISTTTFS()
    mnist.train(test_results_every_epoch=False, valid_results_every_epoch=False)
