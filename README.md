# FedFF
Investigation of the performance of the Forward-Forward algorithm in a federated learning context.

## Simulation parameters
#### Client model parameters:
* LAYER_EPOCHS: &nbsp; training epochs for a single layer
* BIAS_THRESHOLD: &nbsp; model hyper-parameter
* LEARN_RATE: &nbsp; initial learning rate for ADAM optimizer
* WEIGHT_DECAY: &nbsp; weight decay value
* MODEL_UNITS: &nbsp; list containing the number of the units of each layer

#### Federated algorithm parameters:
* MODEL_EPOCHS: &nbsp; training epochs for the client model
* NUM_ROUNDS: &nbsp; number of communication rounds
* NUM_CLIENTS: &nbsp; number of clients in the network
* C_RATE: &nbsp; fraction of selected clients taking part to each round

#### Dataset processing parameters:
* NUM_REPEAT: &nbsp; training dataset repetitions
* BATCH_SIZE: &nbsp; local minibatch size for client update
* SHUFFLE_BUF: &nbsp; dataset shuffle buffer size to sort samples
* PREFETCH_BUF: &nbsp; size of prefetch buffer