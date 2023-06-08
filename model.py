import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.compiler.tf2xla.python import xla
import logging


logger = logging.getLogger()


class FFDense(keras.layers.Layer):
    """
    Custom layer class implementing FF algorihtm logic.
    """

    def __init__(self, units, optimizer, loss_metric, epochs=50, scale=10, use_bias=True,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros",
                 kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing layer object.")
        self.units = units
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)
        self.relu = keras.layers.ReLU()
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.scale = scale
        self.epochs = epochs

    def call(self, inputs):  # normalize and run the input through the dense layer
        x_norm = tf.norm(inputs, ord=2, axis=1, keepdims=True)
        x_norm = x_norm + 1e-5
        x_dir = tf.math.divide(inputs, x_norm)  # consider using tf.math.divide_no_nan
        return self.relu(self.dense(x_dir))

    # core implementation of the FF algorithm
    def forward_forward(self, x_pos, x_neg):
        logger.info("Training layer.")
        for i in range(self.epochs):
            with tf.GradientTape() as tape:
                g_pos = tf.math.reduce_mean(tf.math.pow(self.call(x_pos), 2), 1)
                g_neg = tf.math.reduce_mean(tf.math.pow(self.call(x_neg), 2), 1)
                # Computing SymBa loss.
                loss = tf.math.log(1 + tf.math.exp(-self.scale*tf.math.subtract(g_pos, g_neg)))
                mean_loss = tf.cast(tf.math.reduce_mean(loss), tf.float32)
                self.loss_metric.update_state([mean_loss])
            gradients = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (tf.stop_gradient(self.call(x_pos)),
                tf.stop_gradient(self.call(x_neg)),
                self.loss_metric.result())


class FFNetwork(keras.Model):
    """
    FF model class.
    """

    def __init__(self, units, layer_epochs=50,
                 layer_optimizer=keras.optimizers.legacy.Adam(
                     learning_rate=0.03),
                 **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing model.")
        self.units = units
        self.layer_epochs = layer_epochs
        self.layer_optimizer = layer_optimizer
        self.cumulative_loss = tf.Variable(0.0, trainable=False,
                                           dtype=tf.float32)
        self.loss_count = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.layer_list = [keras.Input(shape=(units[0],))]
        self.layer_list += [FFDense(units[i], optimizer=layer_optimizer, epochs=layer_epochs,
                                    loss_metric=keras.metrics.Mean()) for i in range(1, len(units))]

    def get_config(self):
        return {
            "units": self.units,
            "layer_epochs": self.layer_epochs,
            "layer_optimizer": self.layer_optimizer,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function(reduce_retracing=True)
    def overlay_y_on_x(self, data):
        X_sample, y_sample = data
        max_sample = tf.reduce_max(X_sample, axis=0, keepdims=True)
        max_sample = tf.cast(max_sample, dtype=tf.float64)
        X_zeros = tf.zeros([10], dtype=tf.float64)
        X_update = xla.dynamic_update_slice(X_zeros, max_sample, [y_sample])
        X_sample = xla.dynamic_update_slice(X_sample, X_update, [0])
        return X_sample, y_sample

    @tf.function(reduce_retracing=True)
    def predict_one_sample(self, x):
        goodness_per_label = []
        x = tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1]])
        for label in range(10):
            h, label = self.overlay_y_on_x(data=(x, label))
            h = tf.reshape(h, [-1, tf.shape(h)[0]])
            goodness = []
            for layer_idx in range(1, len(self.layer_list)):
                layer = self.layer_list[layer_idx]
                h = layer(h)
                goodness += [tf.math.reduce_mean(tf.math.pow(h, 2), 1)]
            goodness_per_label += [tf.expand_dims(
                tf.reduce_sum(goodness, keepdims=True), 1)]
        goodness_per_label = tf.concat(goodness_per_label, 1)
        return tf.cast(tf.argmax(goodness_per_label, 1), tf.float64)

    def predict(self, data):
        logger.info("Starting prediction.")
        x = data
        preds = list()
        preds = tf.map_fn(fn=self.predict_one_sample, elems=x)
        return np.asarray(preds, dtype=int)

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data
        x = tf.reshape(x, [-1, tf.shape(x)[1] * tf.shape(x)[2]])  # flatten input
        x_pos, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, y))
        random_y = (y + tf.random.uniform((), 1, 10, tf.int64)) % 10  # mismatched labels
        x_neg, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, random_y))
        h_pos, h_neg = x_pos, x_neg

        for i, layer in enumerate(self.layers):  # TODO: check layer_list
            if isinstance(layer, FFDense):
                logger.info(f"Training dense layer {i+1}.")
                h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
                self.cumulative_loss.assign_add(loss)
                self.loss_count.assign_add(1.0)
                logger.info(f"Done training dense layer {i+1}.")
            else:
                logger.info("Feeding data to the input layer.")
                x = layer(x)

        logger.info("Computing mean loss and updating metrics.")
        mean_res = tf.math.divide(self.cumulative_loss, self.loss_count)
        try:
            tf.debugging.check_numerics(mean_res, "Checking loss validity.")
        except tf.errors.InvalidArgumentError as e:
            logger.critical(e.message)
            raise e
        return {"FinalLoss": mean_res}

    def eval_accuracy(self, data):
        test_samples, test_labels = data
        preds = self.predict(tf.convert_to_tensor(test_samples))
        preds = preds.reshape((preds.shape[0], preds.shape[1]))
        return accuracy_score(preds, test_labels)
