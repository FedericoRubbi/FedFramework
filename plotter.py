import numpy as np
import matplotlib.pyplot as plt
import os
import logging

from config import config, params

from pprint import pprint

logger = logging.getLogger()


def load_data(path=None):
    path = config["resultpath"] if path is None else path
    with open(os.path.join(path, 'accuracy'), 'rb') as f:
        accuracy = np.load(f)
    with open(os.path.join(path, 'loss'), 'rb') as f:
        loss_data = np.load(f, allow_pickle=True)
        loss_data = [np.array_split(
            loss, len(loss) / params["model_epochs"]) if len(loss) else [] for loss in loss_data]
    # with open(os.path.join(path, 'client_rounds'), 'rb') as f:
    #    client_rounds = np.load(f)
    return accuracy, loss_data  # , client_rounds


def plot(path=None):
    path = config["resultpath"] if path is None else path
    if logger:
        logger.info(f"Saving loss and accuracy plots.")
    os.makedirs(path, exist_ok=True)

    #accuracy, loss_data, client_rounds = load_data()
    accuracy, loss_data = load_data(path)
    print(len(set([i for (i, r) in enumerate(loss_data) for l in r if any(np.isnan(l))])))

    with open('log/2023-05-07_21-04-49.log', 'r') as f:
        log = [list(map(int, line[63:-2].split(', '))) for line in f.readlines()
               if 'Updating selected 10 clients' in line]

    # Fix losses with respect to rounds.
    loss = np.zeros([params["num_clients"], params["num_rounds"],
                     params["model_epochs"]])
    # TODO: make loss continuous.
    # for client_i, rounds in enumerate(client_rounds):  # for each client rounds
    #     for loss_i, l in enumerate(loss_data[client_i]):
    #         loss[client_i][rounds[loss_i]] = l

    pprint([(c_i, i) for (c_i, c_l) in enumerate(loss_data) for (i, x) in enumerate(c_l) if any(np.isnan(x))])

    for client_i, l in enumerate(loss_data):  # iterate on each client loss
        loss_i = 0
        # iterate on clients indexes in a round
        for round_i, round_clients in enumerate(log):
            if client_i in round_clients:
                loss[client_i][round_i] = l[loss_i]
                if any(np.isnan(loss[client_i][round_i])) and round_i:
                    loss[client_i][round_i][:] = loss[client_i][round_i - 1][-1]
                loss_i += 1
            elif round_i:  # make loss continuous
                loss[client_i][round_i][:] = loss[client_i][round_i - 1][-1]
    breakpoint()
    loss = [loss.flatten() for loss in loss]
    loss = [np.where(l, l, np.max(l)) for l in loss if any(l)]
    avg_loss, err = np.mean(loss, axis=0), np.std(loss, axis=0)

    breakpoint()

    plt.grid()
    plt.fill_between(np.linspace(0, params["num_rounds"], num=len(avg_loss)),
                     avg_loss - err, avg_loss + err, alpha=0.3)
    for l in loss:
        plt.plot(np.linspace(0, params["num_rounds"], num=len(l)), l,
                 color=(*(np.random.random(2) * 0.5), 0.5))
    plt.plot(np.linspace(0, params["num_rounds"], num=len(avg_loss)),
             avg_loss, color='dodgerblue', label='average loss')
    plt.legend()
    plt.title("Loss over training")
    plt.savefig(os.path.join(path, "loss.png"))
    plt.show()

    plt.grid()
    plt.plot(np.linspace(0, params["num_rounds"], num=len(avg_loss)),
             avg_loss, color='dodgerblue', label='average loss')
    plt.legend()
    plt.title("Loss over training")
    plt.savefig(os.path.join(path, "avg_loss.png"))
    plt.show()

    plt.grid()
    plt.plot(np.linspace(1, params["num_rounds"], num=len(accuracy)),
             accuracy, color='dodgerblue')
    plt.title("Accuracy over training")
    plt.savefig(os.path.join(path, "accuracy.png"))
    plt.show()


if __name__ == "__main__":
    plot(path='simulations/results/2023-05-07_21-04-49')