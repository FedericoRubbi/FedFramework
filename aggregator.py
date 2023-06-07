from tensorflow import keras
import numpy as np
from multiprocessing.pool import ThreadPool
from numpy.random import default_rng
import logging

from config import config, params


RNG = default_rng(seed=1)
logger = logging.getLogger()


class Client:
    """
    Basic client interface for federated algorithms.
    """

    def __init__(self, id, model, dataset, epochs=50):
        self.id = id
        self.model = model
        self.dataset = dataset
        self.dataset_size = sum(dataset.map(
            lambda x, y: len(y)).as_numpy_iterator())
        self.epochs = epochs
        self.history = []
        self.round_cnt = 0
        self.rounds = []
        logger.info(f"Initialized client {id} "
                    f"with dataset size: {self.dataset_size} samples.")

    def update(self, round_index=None, callbacks=[]):
        logger.info(f"Updating client: {self.id}.")
        history = self.model.fit(self.dataset, epochs=self.epochs, callbacks=callbacks)
        logger.info(f"Done updating client: {self.id}.")
        self.history.append(history)
        self.round_cnt += 1
        if round_index is not None:
            self.rounds.append(round_index)

    def log_rounds(self):
        logger.info(f"Client {self.id} was updated in rounds: {', '.join(map(str, self.rounds))}.")


class Server:
    """
    Central server node implementing FedAvg algorithm.
    """

    def __init__(self, clients):
        self.clients = clients
        self.max_clients = len(self.clients)
        self.threaded = config["use_threads"]
        logger.info(f"Initialized server with threaded mode: "
                    f"{'enabled' if self.threaded else 'disabled'}.")

    def execute_round(self, round_index=None):
        round_clients = RNG.choice(
            self.clients, size=max(int(params['c_rate']*self.max_clients), 1), replace=False)
        logger.info(f"Updating selected {len(round_clients)} clients: "
                    f"{', '.join([str(c.id) for c in round_clients])}.")

        if self.threaded:
            with ThreadPool(len(round_clients)) as pool:
                pool.map(lambda client: client.update(round_index), round_clients)
        else:
            for client in round_clients:
                client.update(round_index)
        logger.info("Done updating clients.")

        logger.info("Aggregating updated clients model.")
        models, dataset_sizes = [], []
        for client in round_clients:
            models.append(client.model.get_weights())
            dataset_sizes.append(client.dataset_size)
        new_model = np.average(models, weights=dataset_sizes, axis=0)
        for client in self.clients:  # total partecipation in the broadcast step
            client.model.set_weights(new_model)
        logger.info("Updated model broadcast complete.")

        return round_clients[0].model

    def evaluate_clients(self, test_dataset):
        round_acc = []
        with ThreadPool(len(self.clients)) as pool:
            pool.map(lambda c: round_acc.append(c.model.eval_accuracy(test_dataset)), self.clients)
        avg_accuracy = np.mean(round_acc)
        return avg_accuracy
