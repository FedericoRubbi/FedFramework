import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import json
from numpy.random import default_rng

from config import config, params


RNG = default_rng(seed=1)
logger = logging.getLogger()


def load_data(path=None):
    path = config["resultpath"] if path is None else path
    with open(os.path.join(path, 'loss'), 'rb') as f:
        loss_data = np.load(f, allow_pickle=True)
        loss_data = [np.array_split(
            loss, len(loss) / params["model_epochs"]) if len(loss) else [] for loss in loss_data]
    with open(os.path.join(path, 'avg_accuracy'), 'rb') as f:
        avg_accuracy = np.load(f)
    with open(os.path.join(path, 'accuracy'), 'rb') as f:
        accuracy = np.load(f)
    with open(os.path.join(path, 'client_rounds'), 'rb') as f:
        client_rounds = np.load(f, allow_pickle=True)
    return loss_data, avg_accuracy, accuracy, client_rounds


def plot(path=None, show_plots=False):
    if path is not None:  # load simulation parameters if any
        with open(os.path.join(path, "params.json"), "r") as file:
            sim_params = json.load(file)
            for key in params.keys():
                params[key] = sim_params[key]
    path = config["resultpath"] if path is None else path
    logger.info(f"Plotting loss and accuracy for simulation:\t{path}")
    os.makedirs(path, exist_ok=True)

    loss_data, avg_accuracy, accuracy, client_rounds = load_data(path)
    logger.info("Checking for invalid losses.")
    l = [f'\nclient index: {c_i}\t round index: {r_i}' for (c_i, c_l) in enumerate(loss_data)
         for (r_i, x) in enumerate(c_l) if any(np.isnan(x))]
    if l:
        logger.warning(f"Found {len(l)} invalid losses: {''.join(l)}")
    else:
        logger.info("No invalid loss found.")

    # Fix losses with respect to rounds.
    loss = np.zeros([params["num_clients"], params["num_rounds"], params["model_epochs"]])
    for client_i, rounds in enumerate(client_rounds):
        if not rounds:  # client was never selected
            continue
        loss[client_i][:rounds[0]][:] = loss_data[client_i][0][0]  # write first loss to initial rounds
        for loss_i, l in enumerate(loss_data[client_i]):
            loss[client_i][rounds[loss_i]] = l  # copy round loss to matching position
    loss = [l.flatten() for l in loss]
    for l in loss:  # make loss continuous.
        for i in range(len(l)):
            l[i] = l[i] if l[i] else l[i-1]
    loss = [l for l in loss if any(l)]  # filter never selected client
    avg_loss, err = np.mean(loss, axis=0), np.std(loss, axis=0)

    plt.grid()
    plt.fill_between(np.linspace(0, params["num_rounds"], num=len(avg_loss)),
                     avg_loss - err, avg_loss + err, alpha=0.3)
    for l in RNG.choice(loss, size=3):
        plt.plot(np.linspace(0, params["num_rounds"], num=len(l)), l,
                 color=(*(np.random.random(2) * 0.5), 0.5))
    plt.plot(np.linspace(0, params["num_rounds"], num=len(avg_loss)),
             avg_loss, color='dodgerblue', label='average loss')
    plt.legend()
    plt.title("Loss over training")
    plt.savefig(os.path.join(path, "loss.png"))
    if show_plots:
        plt.show()
    else:
        plt.clf()

    plt.grid()
    plt.plot(np.linspace(0, params["num_rounds"], num=len(avg_loss)), avg_loss, color='dodgerblue')
    plt.title("Loss over training")
    plt.savefig(os.path.join(path, "avg_loss.png"))
    if show_plots:
        plt.show()
    else:
        plt.clf()

    plt.grid()
    plt.plot(np.linspace(1, params["num_rounds"], num=len(accuracy)), accuracy, color='dodgerblue',
             label='accuracy')
    plt.plot(np.linspace(1, params["num_rounds"], num=len(avg_accuracy)), avg_accuracy,
             color='mediumslateblue', label='global accuracy')
    plt.legend()
    plt.title("Accuracy over training")
    plt.savefig(os.path.join(path, "accuracy.png"))
    if show_plots:
        plt.show()
    else:
        plt.clf()

    plt.grid()
    plt.plot(np.linspace(1, params["num_rounds"], num=len(avg_accuracy)), avg_accuracy,
             color='dodgerblue')
    plt.title("Accuracy over training")
    plt.savefig(os.path.join(path, "avg_accuracy.png"))
    if show_plots:
        plt.show()
    else:
        plt.clf()
    logger.info("Done plotting data.")


if __name__ == "__main__":
    plot(path='simulations/results/2023-06-05_14-14-25', show_plots=True)
