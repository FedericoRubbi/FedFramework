# Federated Forward-Forward algorithm
# Author: Federico Rubbi

import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from config import config, params, log_params
from model import FFNetwork
from aggregator import Client, Server
from plotter import plot


logger = logging.getLogger()


def load_datasets():
    logger.info(f"Loading datasets from path: {config['datapath']}.")
    train_datasets, test_datasets = [], []
    for i in range(params["num_clients"]):
        i = ''.join(('00', str(i)))[-3:]
        with open(f'{config["datapath"]}/train/x_train_{i}.npy', 'rb') as fx, \
                open(f'{config["datapath"]}/train/y_train_{i}.npy', 'rb') as fy:
            dataset = tf.data.Dataset.from_tensor_slices((np.load(fx), np.load(fy)))
            train_datasets.append(dataset.repeat(params["num_repeat"])
                                  .shuffle(params["shuffle_buf"], seed=config["seed"])
                                  .batch(params["batch_size"]).prefetch(params["prefetch_buf"]))
    for i in range(params["max_clients"]):
        i = ''.join(('00', str(i)))[-3:]
        with open(f'{config["datapath"]}/test/x_test_{i}.npy', 'rb') as fx, \
                open(f'{config["datapath"]}/test/y_test_{i}.npy', 'rb') as fy:
            test_datasets.append((np.load(fx), np.load(fy)))
    test_samples = np.array([sample for data in test_datasets for sample in data[0]])
    test_labels = np.array([label for data in test_datasets for label in data[1]])
    return train_datasets, (test_samples, test_labels)


def initialize_nodes(train_datasets):
    logger.info("Initializing clients and server nodes.")

    def model_init():
        return FFNetwork(
            units=params["model_units"],
            layer_epochs=params["layer_epochs"],
            layer_optimizer=keras.optimizers.legacy.Adam(
                learning_rate=params["learn_rate"], decay=params["weight_decay"])
        )

    clients = []
    for i in range(params["num_clients"]):
        clients.append(Client(i, model_init(), train_datasets[i], epochs=params["model_epochs"]))
        clients[i].model.compile(jit_compile=True)
    server = Server(clients, threaded=config["use_threads"])
    logger.info("Clients and server nodes initialized")
    return clients, server


def save_data(clients, accuracy, updated_model):
    logger.info(f"Saving metrics and final model.")
    os.makedirs(config["resultpath"], exist_ok=True)
    with open(os.path.join(config["resultpath"], "loss"), "wb+") as f:
        np.save(f, np.array([[l for h in client.history for l in h.history["FinalLoss"]]
                             for client in clients], dtype=object))
    with open(os.path.join(config["resultpath"], "accuracy"), "wb+") as f:
        np.save(f, np.array(accuracy))
    with open(os.path.join(config["resultpath"], "client_rounds"), "wb+") as f:
        np.save(f, np.array([c.rounds for c in clients]))
    for c in clients:
        c.model.save_weights(os.path.join(config["resultpath"], f"model_{c.id}"))
    updated_model.save_weights(os.path.join(
        config["resultpath"], "final_model"))


def main():
    log_params()
    train_datasets, test_dataset = load_datasets()
    clients, server = initialize_nodes(train_datasets)

    accuracy, updated_model = [], None
    for round_i in range(params["num_rounds"]):
        logger.info(f"Running communication round: {round_i}.")
        updated_model = server.execute_round(round_i)
        logger.info("Starting model evaluation.")
        accuracy.append(updated_model.evaluate_accuracy(test_dataset))
        logger.info(f"Evaluated model accuracy: {accuracy[-1]}.")

    for client in clients:
        client.log_rounds()
    save_data(clients, accuracy, updated_model)
    plot()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Exception raised: {repr(e)}")
        raise e