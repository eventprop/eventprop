from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from time import sleep
import logging
from typing import List, Tuple
import os
import gzip
import pickle
import signal
from itertools import cycle

from .layer import Spike, SpikePattern
from .optimizer import GradientDescent, GradientDescentParameters, Optimizer, Adam
from .lif_layer import LIFLayer, LIFLayerParameters
from .loss_layer import TTFSCrossEntropyLoss, TTFSCrossEntropyLossParameters


class TwoLayerTTFS(ABC):
    def __init__(
        self,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(),
        hidden_parameters: LIFLayerParameters = LIFLayerParameters(),
        output_parameters: LIFLayerParameters = LIFLayerParameters(),
        loss_parameters: TTFSCrossEntropyLossParameters = TTFSCrossEntropyLossParameters(),
        weight_increase_threshold_hidden: float = 0.3,
        weight_increase_threshold_output: float = 0.0,
        weight_increase_bump: float = 5e-3,
        lr_decay_gamma: float = 0.95,
        lr_decay_step: int = 2000,
        optimizer: Optimizer = Adam,
    ):
        self.weight_increase_threshold_hidden = weight_increase_threshold_hidden
        self.weight_increase_threshold_output = weight_increase_threshold_output
        self.weight_increase_bump = weight_increase_bump
        self.lr_decay_gamma = lr_decay_gamma
        self.lr_decay_step = lr_decay_step
        self.hidden_parameters = hidden_parameters
        self.output_parameters = output_parameters
        self.loss_parameters = loss_parameters
        self.gd_parameters = gd_parameters
        self.hidden_layer = LIFLayer(self.hidden_parameters)
        self.output_layer = LIFLayer(self.output_parameters)
        self.loss = TTFSCrossEntropyLoss(self.loss_parameters)
        self.optimizer = optimizer(self.loss, self.gd_parameters)
        self.load_data()
        np.random.shuffle(self.train_spikes)
        self._minibatch_idx = 0

    @abstractmethod
    def load_data(self):
        pass

    def save_to_file(self, fname):
        pickle.dump(
            (
                self.losses,
                self.accuracies,
                self.test_accuracies,
                self.test_losses,
                self.valid_accuracies,
                self.valid_losses,
                self.weights,
            ),
            open(fname, "wb"),
        )

    def _get_minibatch(self):
        if self.gd_parameters.batch_size is None:
            return self.train_spikes
        else:
            samples = deepcopy(
                self.train_spikes[
                    self._minibatch_idx : self._minibatch_idx
                    + self.gd_parameters.batch_size
                ]
            )
            self._minibatch_idx += self.gd_parameters.batch_size
            self._minibatch_idx %= len(self.train_spikes)
            return samples

    def valid(self):
        accuracies, losses = list(), list()
        for pattern in self.valid_spikes:
            self.loss(self.output_layer(self.hidden_layer(pattern.spikes)))
            accuracies.append(self.loss.get_classification_result(pattern.label))
            losses.append(self.loss.get_loss(pattern.label))
        logging.debug(f"Got validation accuracy: {np.mean(accuracies)}.")
        logging.debug(f"Got validation loss: {np.mean(losses)}.")
        return np.nanmean(losses), np.mean(accuracies)

    def test(self):
        accuracies, losses = list(), list()
        for pattern in self.test_spikes:
            self.loss(self.output_layer(self.hidden_layer(pattern.spikes)))
            accuracies.append(self.loss.get_classification_result(pattern.label))
            losses.append(self.loss.get_loss(pattern.label))
        logging.debug(f"Got test accuracy: {np.mean(accuracies)}.")
        logging.debug(f"Got test loss: {np.mean(losses)}.")
        return np.nanmean(losses), np.mean(accuracies)

    def train(
        self,
        save_to: str = None,
        save_every: int = 100,
        test_every: int = 100,
        valid_every: int = 100,
    ):
        self.losses, self.accuracies = list(), list()
        self.test_losses, self.test_accuracies = list(), list
        self.valid_accuracies, self.valid_losses = list(), list()
        self.weights = list()
        for epoch in range(self.gd_parameters.iterations):
            minibatch = self._get_minibatch()
            batch_losses, batch_classif_results = list(), list()
            frac_quiet_output, frac_quiet_hidden = list(), list()
            for pattern in minibatch:
                self.loss(self.output_layer(self.hidden_layer(pattern.spikes)))
                self.loss.backward(pattern.label)
                batch_losses.append(self.loss.get_loss(pattern.label))
                batch_classif_results.append(
                    self.loss.get_classification_result(pattern.label)
                )
                frac_quiet_output.append(
                    sum(
                        [len(x) == 0 for x in self.output_layer._post_spikes_per_neuron]
                    )
                    / self.output_layer.parameters.n
                )
                frac_quiet_hidden.append(
                    sum(
                        [len(x) == 0 for x in self.hidden_layer._post_spikes_per_neuron]
                    )
                    / self.hidden_layer.parameters.n
                )
            frac_quiet_output, frac_quiet_hidden = np.mean(frac_quiet_output), np.mean(
                frac_quiet_hidden
            )
            logging.debug(f"Training loss in epoch {epoch}: {np.nanmean(batch_losses)}")
            logging.debug(
                f"Training accuracy in epoch {epoch}: {np.mean(batch_classif_results)}"
            )
            logging.debug(f"Fraction of quiet hidden neurons: {frac_quiet_hidden}")
            logging.debug(f"Fraction of quiet output neurons: {frac_quiet_output}")
            if frac_quiet_hidden > self.weight_increase_threshold_hidden:
                logging.debug("Bumping hidden weights.")
                self.hidden_layer.w_in += self.weight_increase_bump
            else:
                if frac_quiet_output > self.weight_increase_threshold_output:
                    logging.debug("Bumping output weights.")
                    self.output_layer.w_in += self.weight_increase_bump
            self.losses.append(np.nanmean(batch_losses))
            self.accuracies.append(np.mean(batch_classif_results))
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_decay_step is not None and epoch > 0:
                if epoch % self.lr_decay_step == 0:
                    logging.debug(f"Decaying learning rate by {self.lr_decay_gamma}.")
                    self.optimizer.parameters = self.optimizer.parameters._replace(
                        lr=self.optimizer.parameters.lr * self.lr_decay_gamma
                    )
            if valid_every is not None:
                if epoch % valid_every == 0:
                    logging.debug("Getting valid accuracy.")
                    valid_loss, valid_error = self.valid()
                    self.valid_accuracies.append(valid_error)
                    self.valid_losses.append(valid_loss)
                    logging.info(
                        f"Validation accuracy in epoch {epoch}: {valid_error}."
                    )
            if test_every is not None:
                if epoch % test_every == 0:
                    logging.debug("Getting test accuracy.")
                    test_loss, test_error = self.test()
                    self.test_accuracies.append(test_error)
                    self.test_losses.append(test_loss)
                    logging.info(f"Test accuracy in epoch {epoch}: {test_error}.")
            if save_to is not None:
                if epoch % save_every == 0:
                    self.weights.append(
                        (self.hidden_layer.w_in.copy(), self.output_layer.w_in.copy())
                    )
                    logging.debug(f"Saving results to {save_to}.")
                    self.save_to_file(save_to)
        return self.test()
