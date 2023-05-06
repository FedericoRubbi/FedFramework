# Federated Forward-Forward algorithm
# Author: Federico Rubbi

import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from model import FFNetwork
from aggregator import Client, Server

LAYER_EPOCHS = 50
BIAS_THRESHOLD = 10
LEARN_RATE = 0.1
WEIGHT_DECAY = 1e-5
MODEL_UNITS = [784, 1000, 1000, 1000]

MODEL_EPOCHS = 50
NUM_ROUNDS = 50
MAX_CLIENTS = 500
NUM_CLIENTS = 100
C_RATE = 0.1

NUM_REPEAT = 2
BATCH_SIZE = 10  # from FedAvg paper
SHUFFLE_BUF = int(NUM_REPEAT * 0.75 * 100)
PREFETCH_BUF = tf.data.AUTOTUNE

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.join(SCRIPTPATH, 'clients/datasets')
LOGPATH = os.path.join(SCRIPTPATH, 'log', f"{TIMESTAMP}.log")
RESULTPATH = os.path.join(SCRIPTPATH, 'simulations/results', TIMESTAMP)

SEED = 1

logging.basicConfig(level=logging.INFO, filename=LOGPATH, force=True,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def log_params():
    logger.info("Saving current parameters configuration.")
    logger.info(f"- LAYER_EPOCHS:  {LAYER_EPOCHS}")
    logger.info(f"- BIAS_THRESHOLD:  {BIAS_THRESHOLD}")
    logger.info(f"- LEARN_RATE:  {LEARN_RATE}")
    logger.info(f"- WEIGHT_DECAY:  {WEIGHT_DECAY}")
    logger.info(f"- MODEL_UNITS:  {MODEL_UNITS}")

    logger.info(f"- MODEL_EPOCHS:  {MODEL_EPOCHS}")
    logger.info(f"- NUM_ROUNDS:  {NUM_ROUNDS}")
    logger.info(f"- MAX_CLIENTS:  {MAX_CLIENTS}")
    logger.info(f"- NUM_CLIENTS:  {NUM_CLIENTS}")
    logger.info(f"- C_RATE:  {C_RATE}")

    logger.info(f"- NUM_REPEAT:  {NUM_REPEAT}")
    logger.info(f"- BATCH_SIZE:  {BATCH_SIZE}")
    logger.info(f"- SHUFFLE_BUF:  {SHUFFLE_BUF}")
    logger.info(f"- PREFETCH_BUF:  {PREFETCH_BUF}")


def load_datasets():
    logger.info(f"Loading datasets from path: {DATAPATH}.")
    train_datasets, test_datasets = [], []
    for i in range(NUM_CLIENTS):
        i = ''.join(('00', str(i)))[-3:]
        with open(f'{DATAPATH}/train/x_train_{i}.npy', 'rb') as fx, \
                open(f'{DATAPATH}/train/y_train_{i}.npy', 'rb') as fy:
            dataset = tf.data.Dataset.from_tensor_slices((np.load(fx),
                                                          np.load(fy)))
            train_datasets.append(dataset.repeat(NUM_REPEAT)
                                  .shuffle(SHUFFLE_BUF, seed=SEED)
                                  .batch(BATCH_SIZE).prefetch(PREFETCH_BUF))
    for i in range(MAX_CLIENTS):
        i = ''.join(('00', str(i)))[-3:]
        with open(f'{DATAPATH}/test/x_test_{i}.npy', 'rb') as fx, \
                open(f'{DATAPATH}/test/y_test_{i}.npy', 'rb') as fy:
            test_datasets.append((np.load(fx), np.load(fy)))
    test_samples = np.array(
        [sample for data in test_datasets for sample in data[0]])
    test_labels = np.array(
        [label for data in test_datasets for label in data[1]])
    return train_datasets, (test_samples, test_labels)


def initialize_nodes(train_datasets):
    logger.info("Initializing clients and server nodes.")

    def model_init():
        return FFNetwork(
            units=MODEL_UNITS,
            layer_epochs=LAYER_EPOCHS,
            layer_optimizer=keras.optimizers.legacy.Adam(
                learning_rate=LEARN_RATE, decay=WEIGHT_DECAY)
        )

    clients = []
    for i in range(NUM_CLIENTS):
        clients.append(Client(i, model_init(), train_datasets[i],
                              epochs=MODEL_EPOCHS))
        clients[i].model.compile(jit_compile=True)
    server = Server(clients, threaded=True)
    logger.info("Clients and server nodes initialized")
    return clients, server


def save_plots(clients, accuracy):
    logger.info(f"Saving loss and accuracy plots.")
    os.makedirs(RESULTPATH, exist_ok=True)
    l = np.array([[loss for h in client.history for loss in h.history["FinalLoss"]]
                  for client in clients], dtype=object)

    def tolerant_mean(arrs):
        lens = [len(i) for i in arrs]
        arr = np.ma.empty((np.max(lens), len(arrs)))
        arr.mask = True
        for idx, l in enumerate(arrs):
            arr[:len(l), idx] = l
        return arr.mean(axis=-1), arr.std(axis=-1)

    y, error = tolerant_mean(l)
    plt.grid()
    plt.fill_between(np.linspace(1, NUM_ROUNDS, num=len(y)),
                     y - error, y + error, alpha=0.3)
    for loss in l:
        plt.plot(np.linspace(1, NUM_ROUNDS, num=len(loss)),
                 loss, color=(*np.random.random(2), 0.5))
    plt.plot(np.linspace(1, NUM_ROUNDS, num=len(y)), y, color='dodgerblue')
    plt.title("Loss over training")
    plt.savefig(os.path.join(RESULTPATH, "loss.png"))
    plt.clf()

    y, error = tolerant_mean(l)
    plt.grid()
    plt.plot(np.linspace(1, NUM_ROUNDS, num=len(y)), y, color='dodgerblue')
    plt.title("Loss over training")
    plt.savefig(os.path.join(RESULTPATH, "avg_loss.png"))
    plt.clf()

    plt.grid()
    plt.plot(np.linspace(1, NUM_ROUNDS, num=len(accuracy)),
             accuracy, color='dodgerblue')
    plt.title("Accuracy over training")
    plt.savefig(os.path.join(RESULTPATH, "accuracy.png"))
    plt.clf()


def save_data(clients, accuracy, updated_model):
    logger.info(f"Saving metrics and final model.")
    os.makedirs(RESULTPATH, exist_ok=True)
    with open(os.path.join(RESULTPATH, "loss"), "wb+") as f:
        np.save(f, np.array([[loss for h in client.history for loss in h.history["FinalLoss"]]
                             for client in clients], dtype=object))
    with open(os.path.join(RESULTPATH, "accuracy"), "wb+") as f:
        np.save(f, np.array(accuracy))
    for c in clients:
        c.model.save_weights(os.path.join(RESULTPATH, f"model_{c.id}"))
    updated_model.save_weights(os.path.join(RESULTPATH, "final_model"))


def main():
    log_params()
    train_datasets, test_dataset = load_datasets()
    clients, server = initialize_nodes(train_datasets)

    accuracy, updated_model = [], None
    for round_i in range(NUM_ROUNDS):
        logger.info(f"Running communication round {round_i}.")
        updated_model = server.execute_round()
        logger.info("Starting model evaluation.")
        accuracy.append(updated_model.evaluate_accuracy(test_dataset))
        logger.info(f"Evaluated model accuracy: {accuracy[-1]}.")

    save_plots(clients, accuracy)
    save_data(clients, accuracy, updated_model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Exception raised: {repr(e)}")
        raise e
