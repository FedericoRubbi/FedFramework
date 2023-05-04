from tensorflow import keras
import numpy as np
from multiprocessing.pool import ThreadPool
from numpy.random import default_rng
import logging


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
        logger.info(f"Initialized client {id} "
                    f"with dataset size: {self.dataset_size} samples.")

    def update(self):
        logger.info(f"Updating client: {self.id}.")
        history = self.model.fit(self.dataset, epochs=self.epochs)
        logger.info(f"Done updating client: {self.id}.")
        self.history.append(history)
        self.round_cnt += 1


class Server:
    """
    Central server node implementing FedAvg algorithm.
    """

    def __init__(self, clients, threaded=False):
        self.clients = clients
        self.max_clients = len(self.clients)
        self.threaded = threaded
        logger.info(f"Initialized server with threaded mode: "
                    f"{'enabled' if threaded else 'disabled'}.")

    def execute_round(self):
        round_clients = default_rng().choice(
            self.clients, size=max(int(0.1*self.max_clients), 1), replace=False)
        logger.info(f"Updating selected {len(round_clients)} clients: "
                    f"{', '.join([str(c.id) for c in round_clients])}.")

        if self.threaded:
            with ThreadPool(len(round_clients)) as pool:
                pool.map(lambda client: client.update(), round_clients)
        else:
            for client in round_clients:
                client.update()
        logger.info("Done updating clients.")

        logger.info("Aggregating updated clients model.")
        models, dataset_sizes = [], []
        for client in round_clients:
            models.append(client.model.get_weights())
            dataset_sizes.append(client.dataset_size)
        new_model = np.average(models, weights=dataset_sizes, axis=0)
        for client in round_clients:
            client.model.set_weights(new_model)
        logger.info("Updated model broadcast complete.")

        return round_clients[0].model
