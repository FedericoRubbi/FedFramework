import sys
import os
import json
import logging
from datetime import datetime
import tensorflow as tf


config = {
    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "scriptpath": os.path.dirname(os.path.realpath(sys.argv[0])),
    "seed": 1,
    "use_threads": True,
}
config["datapath"] = os.path.join(config["scriptpath"], 'clients/datasets')
config["logpath"] = os.path.join(config["scriptpath"], 'log', config["timestamp"] + ".log")
config["resultpath"] = os.path.join(config["scriptpath"], 'simulations/results', config["timestamp"])

logging.basicConfig(level=logging.INFO, filename=config["logpath"], force=True,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
with open(os.path.join(config["scriptpath"], "params.json"), "r") as file:
    params = json.load(file)

if params["shuffle_buf"] is None:
    params["shuffle_buf"] = int(params["num_repeat"] * 0.75 * 100)
if params["prefetch_buf"] is None:
    params["prefetch_buf"] = tf.data.AUTOTUNE


def log_params():
    logger.info("Saving current parameters configuration.")
    logger.info(f"- layer_epochs:  {params['layer_epochs']}")
    logger.info(f"- scale_factor:  {params['scale_factor']}")
    logger.info(f"- learn_rate:  {params['learn_rate']}")
    logger.info(f"- weight_decay:  {params['weight_decay']}")
    logger.info(f"- model_units:  {params['model_units']}")
    logger.info(f"- model_epochs:  {params['model_epochs']}")
    logger.info(f"- num_rounds:  {params['num_rounds']}")
    logger.info(f"- max_clients:  {params['max_clients']}")
    logger.info(f"- num_clients:  {params['num_clients']}")
    logger.info(f"- c_rate:  {params['c_rate']}")
    logger.info(f"- num_repeat:  {params['num_repeat']}")
    logger.info(f"- batch_size:  {params['batch_size']}")
    logger.info(f"- shuffle_buf:  {params['shuffle_buf']}")
    logger.info(f"- prefetch_buf:  {params['prefetch_buf']}")
