from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import logging
from typing import NamedTuple, Tuple, Iterator
import pickle

from .optimizer import GradientDescentParameters, Optimizer, Adam
from .layer import Layer, SpikeDataset


class AbstractTraining(ABC):
    @abstractmethod
    def __init__(
        self,
        loss_class: Layer,
        loss_parameters: NamedTuple,
        gd_parameters: GradientDescentParameters = GradientDescentParameters(),
        weight_increase_bump: float = 5e-3,
        lr_decay_gamma: float = 0.95,
        lr_decay_step: int = 2000,
        optimizer_class: Optimizer = Adam,
    ):
        self.weight_increase_bump = weight_increase_bump
        self.lr_decay_gamma = lr_decay_gamma
        self.lr_decay_step = lr_decay_step
        self.loss_class = loss_class
        self.loss_parameters = loss_parameters
        self.gd_parameters = gd_parameters
        self.loss = self.loss_class(self.loss_parameters)
        self.optimizer = optimizer_class(self.loss, self.gd_parameters)
        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    def _get_minibatch(self) -> Iterator[SpikeDataset]:
        logging.debug("Shuffling training data.")
        self.train_batch.shuffle()
        if self.gd_parameters.minibatch_size is None:
            yield self.train_batch
        else:
            minibatch_idx = 0
            while minibatch_idx < len(self.train_batch):
                yield self.train_batch[
                    minibatch_idx : minibatch_idx + self.gd_parameters.minibatch_size
                ]
                minibatch_idx += self.gd_parameters.minibatch_size

    def _get_results_for_set(self, dataset: SpikeDataset) -> Tuple[float, float]:
        self.forward(dataset)
        accuracy = self.loss.get_accuracy(dataset.labels)
        losses = self.loss.get_losses(dataset.labels)
        logging.debug(f"Got accuracy: {accuracy}.")
        logging.debug(f"Got loss: {np.mean(losses)}.")
        return np.nanmean(losses), accuracy

    def valid(self) -> Tuple[float, float]:
        valid_loss, valid_accuracy = self._get_results_for_set(self.valid_batch)
        self.valid_accuracies.append(valid_accuracy)
        self.valid_losses.append(valid_loss)
        return valid_loss, valid_accuracy

    def test(self) -> Tuple[float, float]:
        test_loss, test_accuracy = self._get_results_for_set(self.test_batch)
        self.test_accuracies.append(test_accuracy)
        self.test_losses.append(test_loss)
        return test_loss, test_accuracy

    def save_to_file(self, fname: str):
        pickle.dump(
            self.get_data_for_pickling(),
            open(fname, "wb"),
        )

    def get_data_for_pickling(self):
        return (
            self.losses,
            self.accuracies,
            self.test_accuracies,
            self.test_losses,
            self.valid_accuracies,
            self.valid_losses,
            self.weights,
        )

    def reset_results(self):
        self.losses, self.accuracies = list(), list()
        self.test_losses, self.test_accuracies = list(), list()
        self.valid_accuracies, self.valid_losses = list(), list()
        self.weights = list()

    def forward_and_backward(self, minibatch: SpikeDataset):
        self.forward(minibatch)
        self.backward(minibatch)

    @abstractmethod
    def forward(self, minibatch: SpikeDataset):
        pass

    @abstractmethod
    def backward(self, minibatch: SpikeDataset):
        pass

    @abstractmethod
    def process_dead_neurons(self):
        pass

    @abstractmethod
    def get_weight_copy(self) -> Tuple:
        pass

    def train(
        self,
        save_to: str = None,
        save_every: int = None,
        save_final_weights_only: bool = False,
        train_results_every_minibatch: bool = True,
        test_results_every_epoch: bool = False,
        valid_results_every_epoch: bool = False,
    ):
        self.reset_results()
        for epoch in range(self.gd_parameters.epochs):
            if valid_results_every_epoch:
                logging.debug("Getting valid accuracy.")
                self.valid()
                logging.info(
                    f"Validation accuracy in epoch {epoch}: {self.valid_accuracies[-1]}."
                )
            if test_results_every_epoch:
                logging.debug("Getting test accuracy.")
                self.test()
                logging.info(
                    f"Test accuracy in epoch {epoch}: {self.test_accuracies[-1]}."
                )
            for mb_idx, minibatch in enumerate(self._get_minibatch()):
                self.forward_and_backward(minibatch)
                if train_results_every_minibatch is not None:
                    batch_loss = np.nanmean(self.loss.get_losses(minibatch.labels))
                    batch_accuracy = self.loss.get_accuracy(minibatch.labels)
                    logging.debug(
                        f"Training loss in epoch {epoch}, minibatch {mb_idx}: {batch_loss}"
                    )
                    logging.debug(
                        f"Training accuracy in epoch {epoch}, minibatch {mb_idx}: {batch_accuracy}"
                    )
                    self.losses.append(batch_loss)
                    self.accuracies.append(batch_accuracy)
                self.process_dead_neurons()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.lr_decay_step is not None and epoch > 0:
                if epoch % self.lr_decay_step == 0:
                    logging.debug(f"Decaying learning rate by {self.lr_decay_gamma}.")
                    self.optimizer.parameters = self.optimizer.parameters._replace(
                        lr=self.optimizer.parameters.lr * self.lr_decay_gamma
                    )
            if save_to is not None:
                if epoch % save_every == 0:
                    if not save_final_weights_only:
                        self.weights.append(self.get_weight_copy())
                    logging.debug(f"Saving results to {save_to}.")
                    self.save_to_file(save_to)
        if save_to is not None:
            self.weights.append(self.get_weight_copy())
            logging.debug(f"Saving results to {save_to}.")
            self.save_to_file(save_to)
        return self.test()


class AbstractOneLayer(AbstractTraining):
    @abstractmethod
    def __init__(
        self,
        output_layer_class: Layer,
        output_parameters: NamedTuple,
        *args,
        weight_increase_threshold_output: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_increase_threshold_output = weight_increase_threshold_output
        self.output_parameters = output_parameters
        self.output_layer = output_layer_class(self.output_parameters)

    def forward(self, minibatch: SpikeDataset):
        self.loss(self.output_layer(minibatch.spikes))

    def backward(self, minibatch: SpikeDataset):
        self.loss.backward(minibatch.labels)

    def process_dead_neurons(self):
        frac_quiet_output = self.output_layer.dead_fraction
        logging.debug(f"Fraction of quiet output neurons: {frac_quiet_output}")
        if frac_quiet_output > self.weight_increase_threshold_output:
            logging.debug("Bumping output weights.")
            self.output_layer.w_in += self.weight_increase_bump

    def get_weight_copy(self) -> Tuple:
        return self.output_layer.w_in.copy()


class AbstractTwoLayer(AbstractTraining):
    @abstractmethod
    def __init__(
        self,
        hidden_layer_class: Layer,
        output_layer_class: Layer,
        hidden_parameters: NamedTuple,
        output_parameters: NamedTuple,
        *args,
        weight_increase_threshold_hidden: float = 0.3,
        weight_increase_threshold_output: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_increase_threshold_hidden = weight_increase_threshold_hidden
        self.weight_increase_threshold_output = weight_increase_threshold_output
        self.hidden_parameters = hidden_parameters
        self.output_parameters = output_parameters
        self.hidden_layer_class = hidden_layer_class
        self.output_layer_class = output_layer_class
        self.hidden_layer = self.hidden_layer_class(self.hidden_parameters)
        self.output_layer = self.output_layer_class(self.output_parameters)

    def forward(self, minibatch: SpikeDataset):
        self.loss(self.output_layer(self.hidden_layer(minibatch.spikes)))

    def backward(self, minibatch: SpikeDataset):
        self.loss.backward(minibatch.labels)

    def process_dead_neurons(self):
        frac_quiet_output = self.output_layer.dead_fraction
        frac_quiet_hidden = self.hidden_layer.dead_fraction
        logging.debug(f"Fraction of quiet hidden neurons: {frac_quiet_hidden}")
        logging.debug(f"Fraction of quiet output neurons: {frac_quiet_output}")
        if frac_quiet_hidden > self.weight_increase_threshold_hidden:
            logging.debug("Bumping hidden weights.")
            self.hidden_layer.w_in += self.weight_increase_bump
        else:
            if frac_quiet_output > self.weight_increase_threshold_output:
                logging.debug("Bumping output weights.")
                self.output_layer.w_in += self.weight_increase_bump

    def get_weight_copy(self) -> Tuple:
        return (self.hidden_layer.w_in.copy(), self.output_layer.w_in.copy())