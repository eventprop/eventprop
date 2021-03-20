import numpy as np
from tqdm import tqdm
import pickle
import h5py
import logging
import os
import zipfile
import urllib.request

from eventprop.eventprop_cpp import Spikes, SpikesVector
from eventprop.layer import GaussianDistribution, SpikeDataset, UniformDistribution
from eventprop.lif_layer import LIFLayerParameters
from eventprop.vmax_training import (
    OneLayerVMax,
    TwoLayerVMax,
    VMaxCrossEntropyLossParameters,
)
from eventprop.optimizer import GradientDescentParameters

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "digits_dataset")
train_path_raw = os.path.join(dir_path, "shd_train_h5.zip")
test_path_raw = os.path.join(dir_path, "shd_test_h5.zip")
train_path = os.path.join(dir_path, "shd_train_h5.pkl")
test_path = os.path.join(dir_path, "shd_test_h5.pkl")


class DigitsMixin:
    def load_data(self):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            self._convert_data()
        self.train_batch = pickle.load(open(train_path, "rb"))
        self.test_batch = pickle.load(open(test_path, "rb"))

    def _convert_data(self):
        if not os.path.exists(train_path_raw):
            logging.info("Downloading training dataset...")
            urllib.request.urlretrieve(
                "https://compneuro.net/datasets/shd_train.h5.zip", train_path_raw
            )
        if not os.path.exists(test_path_raw):
            logging.info("Downloading test dataset...")
            urllib.request.urlretrieve(
                "https://compneuro.net/datasets/shd_test.h5.zip", test_path_raw
            )

        train_hdf5 = h5py.File(
            zipfile.ZipFile(train_path_raw).open("shd_train.h5"), "r"
        )
        test_hdf5 = h5py.File(zipfile.ZipFile(test_path_raw).open("shd_test.h5"), "r")
        train_labels = np.array(train_hdf5["labels"])
        train_times = train_hdf5["spikes"]["times"]
        train_units = train_hdf5["spikes"]["units"]
        test_labels = np.array(test_hdf5["labels"])
        test_times = test_hdf5["spikes"]["times"]
        test_units = test_hdf5["spikes"]["units"]

        logging.info("Converting train dataset...")
        train_patterns = list()
        for times, units in tqdm(zip(train_times, train_units)):
            train_patterns.append(
                Spikes(
                    np.array(times, dtype=np.float64), np.array(units, dtype=np.int32)
                )
            )
        train_batch = SpikeDataset(SpikesVector(train_patterns), train_labels)
        with open(train_path, "wb") as f:
            pickle.dump(train_batch, f)

        logging.info("Converting test dataset...")
        test_patterns = list()
        for times, units in tqdm(zip(test_times, test_units)):
            test_patterns.append(
                Spikes(
                    np.array(times, dtype=np.float64), np.array(units, dtype=np.int32)
                )
            )
        test_batch = SpikeDataset(SpikesVector(test_patterns), test_labels)
        with open(test_path, "wb") as f:
            pickle.dump(test_batch, f)


class OneLayerDigitsVMax(DigitsMixin, OneLayerVMax):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(
            minibatch_size=256, iterations=30000, lr=1e-3, gradient_clip=None
        ),
        output_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=700,
            n=128,
            tau_mem=20e-3,
            tau_syn=5e-3,
            w_dist=GaussianDistribution(
                w_mean=0.25 * 1 / np.sqrt(700), w_std=0.25 * 1 / np.sqrt(700)
            ),
        ),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(
            n=20,
            n_in=128,
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


class TwoLayerDigitsVMax(DigitsMixin, TwoLayerVMax):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(
            minibatch_size=256, iterations=30000, lr=1e-3, gradient_clip=None
        ),
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=700,
            n=128,
            tau_mem=20e-3,
            tau_syn=10e-3,
            w_dist=GaussianDistribution(
                w_mean=0.25 * 1 / np.sqrt(700), w_std=0.25 * 1 / np.sqrt(700), seed=0
            )
            # w_dist=UniformDistribution(
            #    w_lower=-1 / np.sqrt(700), w_upper=1 / np.sqrt(700)
            # ),
        ),
        output_parameters: LIFLayerParameters = LIFLayerParameters(
            n_in=128,
            n=20,
            tau_mem=20e-3,
            tau_syn=10e-3,
            w_dist=GaussianDistribution(
                w_mean=0.25 * 1 / np.sqrt(128), w_std=0.25 * 1 / np.sqrt(128), seed=0
            )
            # w_dist=UniformDistribution(
            #    w_lower=-1 / np.sqrt(128), w_upper=1 / np.sqrt(128)
            # ),
        ),
        loss_parameters: VMaxCrossEntropyLossParameters = VMaxCrossEntropyLossParameters(
            n=20,
            n_in=20,
            w_dist=GaussianDistribution(
                w_mean=0.25 * 1 / np.sqrt(20), w_std=0.25 * 1 / np.sqrt(20), seed=0
            )
            # w_dist=UniformDistribution(
            #    w_lower=-1 / np.sqrt(20), w_upper=1 / np.sqrt(20)
            # ),
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dvmax = OneLayerDigitsVMax(
        weight_increase_bump=1e-3, weight_increase_threshold_output=0.2
    )
    dvmax.train(test_every=None, valid_every=None)