import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling1D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, Concatenate, GlobalAveragePooling1D, TimeDistributed
from tensorflow.keras.layers import Attention
from tensorflow import keras

n_neurons = 96  # Number of neurons in a hidden layer in the base neural network


def triplet_loss_func(y_true, y_pred, alpha=0.2):
    global n_neurons
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
                    anchor -- the encodings for the anchor data
                    positive -- the encodings for the positive data (similar to anchor)
                    negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    # The input y_pred a concatenation of anchor, positive, and negative embeddings.
    # Each embedding has a length of n_neurons.
    # The total width of y_pred is therefore 3 * n_neuron

    anchor = y_pred[:, 0:n_neurons]
    positive = y_pred[:, n_neurons:2*n_neurons]
    negative = y_pred[:, 2*n_neurons:3*n_neurons]

    # euclidean
    # distance between the anchor and the positive
    pos_dist = tf.math.reduce_sum(
        tf.math.square(anchor-positive), axis=1)
    # distance between the anchor and the negative
    neg_dist = tf.math.reduce_sum(
        tf.math.square(anchor-negative), axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = tf.math.maximum(
        basic_loss, 0.0)  # positive or zero loss

    return loss


def triplet_learning(use_base, length):
    # code from #https://stackoverflow.com/questions/53576436/predicting-image-using-triplet-loss with modifications
    positive_example = Input(shape=(length, 1))
    negative_example = Input(shape=(length, 1))
    anchor_example = Input(shape=(length, 1))

    # embeddings
    positive_embedding = use_base(positive_example)
    negative_embedding = use_base(negative_example)
    anchor_embedding = use_base(anchor_example)

    merged_output = tf.keras.layers.concatenate(
        [anchor_embedding, positive_embedding, negative_embedding])
    model = Model(inputs=[anchor_example, positive_example, negative_example],
                  outputs=merged_output)
    return model


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv1D(
            filters, kernel_size, activation="relu", padding="same")
        self.conv2 = keras.layers.Conv1D(
            filters, kernel_size, activation=None, padding="same")
        self.activation = keras.layers.Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return self.activation(x + inputs)  # Adding the residual connection


def baseCNN(input_dim: int, hidden_dim=64):
    return keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim, 1)),

        # First convolutional block
        keras.layers.Conv1D(hidden_dim, kernel_size=7,
                            activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),  # Downsampling

        # Second convolutional block
        keras.layers.Conv1D(hidden_dim * 2, kernel_size=5,
                            activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),  # Downsampling

        # Third convolutional block
        keras.layers.Conv1D(hidden_dim * 4, kernel_size=3,
                            activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),  # Downsampling

        # Fourth convolutional block
        keras.layers.Conv1D(hidden_dim * 8, kernel_size=3,
                            activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),  # Downsampling

        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(n_neurons * 2, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(n_neurons, activation='relu')
    ])


def baseNN(length: int):
    inputted = Input(shape=(length,))
    x = Dense(n_neurons, activation='relu')(inputted)
    x = Dropout(0.1)(x)
    x = Dense(n_neurons, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(n_neurons, activation='relu')(x)
    x = Dropout(0.1)(x)
    base_network = Model(inputs=inputted, outputs=x)
    return base_network


def baseLSTM(length: int):
    inputted = Input(shape=(length, 1))
    x = LSTM(n_neurons, recurrent_dropout=0.1)(inputted)
    base_network = Model(inputs=inputted, outputs=x)
    return base_network


def baseGRU(length: int):
    inputted = Input(shape=(length, 1))
    x = GRU(n_neurons, recurrent_dropout=0.1)(inputted)
    base_network = Model(inputs=inputted, outputs=x)
    return base_network


def baseBiGRU(length: int):
    inputted = Input(shape=(length, 1))
    x = Bidirectional(GRU(n_neurons))(inputted)
    x = Dropout(0.1)(x)
    base_network = Model(inputs=inputted, outputs=x)
    return base_network


def complexGRU(length: int):
    inputted = Input(shape=(length, 1))
    x = GRU(length, recurrent_dropout=0.1, return_sequences=True)(inputted)
    x = GRU(n_neurons * 2, recurrent_dropout=0.1, return_sequences=True)(x)
    x = GRU(n_neurons, recurrent_dropout=0.1, return_sequences=False)(x)

    # x = Dense(n_neurons, activation='relu')(x)
    # x = Dropout(0.1)(x)
    base_network = Model(inputs=inputted, outputs=x)
    return base_network


def baseBiLSTM(length: int):
    inputted = Input(shape=(length, 1))
    x = Bidirectional(LSTM(n_neurons))(inputted)
    x = Dropout(0.1)(x)
    base_network = Model(inputs=inputted, outputs=x)
    return base_network
