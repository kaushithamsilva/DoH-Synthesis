import tensorflow as tf
from tensorflow import keras
import numpy as np

import pandas as pd
import sys


@tf.keras.utils.register_keras_serializable()
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@tf.keras.utils.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Reshape((1, input_dim, )),
            keras.layers.GRU(hidden_dim, return_sequences=True,
                             recurrent_dropout=0.1),
            keras.layers.GRU(hidden_dim, return_sequences=False,
                             recurrent_dropout=0.1),
        ])

        # Latent space layers
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

        # Decoder with GRU layers
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(hidden_dim, activation="relu"),
            keras.layers.RepeatVector(1),  # Prepare for GRU layers
            keras.layers.GRU(hidden_dim, return_sequences=True,
                             activation="relu"),
            keras.layers.GRU(input_dim, return_sequences=False,
                             activation="relu"),
            keras.layers.Dense(input_dim, activation='linear')  # Output layer
        ])

    def encode(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder(z)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DenseVAE(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        super(DenseVAE, self).__init__(
            input_dim, latent_dim, hidden_dim, **kwargs)

        # Store new hyperparameters for config
        self.dropout_rate = 0.2
        self.l2_reg = 0.01

        # Enhanced encoder with regularization
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),

            # First dense block
            keras.layers.Dense(
                hidden_dim,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                bias_regularizer=keras.regularizers.l2(self.l2_reg)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.dropout_rate),

            # Second dense block
            keras.layers.Dense(
                hidden_dim // 2,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                bias_regularizer=keras.regularizers.l2(self.l2_reg)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.dropout_rate),

            # Third dense block
            keras.layers.Dense(
                hidden_dim // 4,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                bias_regularizer=keras.regularizers.l2(self.l2_reg)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.dropout_rate),
        ])

        # Latent space layers with regularization
        self.z_mean = keras.layers.Dense(
            latent_dim,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )
        self.z_log_var = keras.layers.Dense(
            latent_dim,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )
        self.sampling = Sampling()

        # Enhanced decoder with regularization
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),

            # First dense block
            keras.layers.Dense(
                hidden_dim // 4,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                bias_regularizer=keras.regularizers.l2(self.l2_reg)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.dropout_rate),

            # Second dense block
            keras.layers.Dense(
                hidden_dim // 2,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                bias_regularizer=keras.regularizers.l2(self.l2_reg)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.dropout_rate),

            # Third dense block
            keras.layers.Dense(
                hidden_dim,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                bias_regularizer=keras.regularizers.l2(self.l2_reg)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.dropout_rate),

            # Output layer
            keras.layers.Dense(
                input_dim,
                activation='linear',
                kernel_regularizer=keras.regularizers.l2(self.l2_reg)
            )
        ])


@tf.keras.utils.register_keras_serializable()
class LSTM_VAE(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        super(LSTM_VAE, self).__init__(
            input_dim, latent_dim, hidden_dim, **kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Reshape((1, input_dim, )),
            keras.layers.LSTM(int(hidden_dim * 1.5), return_sequences=True,
                              recurrent_dropout=0.1),
            keras.layers.LSTM(hidden_dim, return_sequences=False),
        ])

        # Latent space layers
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

        # Decoder with GRU layers
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(hidden_dim, activation="relu"),
            keras.layers.RepeatVector(1),  # Prepare for GRU layers
            keras.layers.LSTM(int(hidden_dim * 1.5), return_sequences=False),
            keras.layers.Dense(input_dim)  # Output layer
        ])


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


@tf.keras.utils.register_keras_serializable()
class ConvVAE_NoSkips(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        super(ConvVAE, self).__init__(
            input_dim, latent_dim, hidden_dim, **kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder with residual blocks
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim, 1)),
            keras.layers.Conv1D(hidden_dim, kernel_size=8,
                                strides=2, activation="relu", padding="same"),
            # ResidualBlock(hidden_dim, kernel_size=3),  # First residual block
            keras.layers.Conv1D(hidden_dim * 2, kernel_size=8,
                                strides=2, activation="relu", padding="same"),
            # Second residual block
            # ResidualBlock(hidden_dim * 2, kernel_size=3),
            keras.layers.Flatten(),
        ])

        # Latent space layers
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

        # Decoder with residual blocks
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(
                hidden_dim * 2 * (128 // 4), activation="relu"),
            keras.layers.Reshape(
                (128 // 4, hidden_dim * 2, )),  # Reshape to 3D
            keras.layers.Conv1DTranspose(
                hidden_dim * 2, kernel_size=3, strides=2, activation="relu", padding="same"),
            # First residual block in decoder
            # ResidualBlock(hidden_dim * 2, kernel_size=3),
            keras.layers.Conv1DTranspose(
                hidden_dim, kernel_size=3, strides=2, activation="relu", padding="same"),
            # Second residual block in decoder
            # ResidualBlock(hidden_dim, kernel_size=3),
            keras.layers.Conv1DTranspose(1, kernel_size=3, padding="same"),
            keras.layers.Reshape((128,)),
            keras.layers.Dense(input_dim, activation='relu'),
            keras.layers.Dense(input_dim)
        ])


# Use the below for models before 15/11/2024
@tf.keras.utils.register_keras_serializable()
class ConvVAE(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        super(ConvVAE, self).__init__(
            input_dim, latent_dim, hidden_dim, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Encoder with residual blocks
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim, 1)),
            keras.layers.Conv1D(hidden_dim, kernel_size=7,
                                strides=2, activation="relu", padding="same"),
            ResidualBlock(hidden_dim, kernel_size=5),  # First residual block
            keras.layers.Conv1D(hidden_dim * 2, kernel_size=3,
                                strides=2, activation="relu", padding="same"),
            # Second residual block
            ResidualBlock(hidden_dim * 2, kernel_size=3),
            keras.layers.Flatten(),
        ])
        # Latent space layers
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()
        # Decoder with residual blocks
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(
                hidden_dim * 2 * (128 // 4), activation="relu"),
            keras.layers.Reshape(
                (128 // 4, hidden_dim * 2, )),  # Reshape to 3D
            keras.layers.Conv1DTranspose(
                hidden_dim * 2, kernel_size=3, strides=2, activation="relu", padding="same"),
            # First residual block in decoder
            ResidualBlock(hidden_dim * 2, kernel_size=3),
            keras.layers.Conv1DTranspose(
                hidden_dim, kernel_size=3, strides=2, activation="relu", padding="same"),
            # Second residual block in decoder
            ResidualBlock(hidden_dim, kernel_size=5),
            keras.layers.Conv1DTranspose(1, kernel_size=7, padding="same"),
            keras.layers.Reshape((128,)),
            keras.layers.Dense(input_dim, activation='relu'),
            keras.layers.Dense(input_dim)
        ])


@tf.keras.utils.register_keras_serializable()
class ConvVAE_BatchNorm(VAE):
    def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
        super(ConvVAE_BatchNorm, self).__init__(
            input_dim, latent_dim, hidden_dim, **kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Encoder with residual blocks
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim, 1)),
            keras.layers.Conv1D(hidden_dim, kernel_size=7,
                                strides=2, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            ResidualBlock(hidden_dim, kernel_size=5),  # First residual block
            keras.layers.Conv1D(hidden_dim * 2, kernel_size=3,
                                strides=2, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            # Second residual block
            ResidualBlock(hidden_dim * 2, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
        ])
        # Latent space layers
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()
        # Decoder with residual blocks
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(
                hidden_dim * 2 * (128 // 4), activation="relu"),
            keras.layers.Reshape(
                (128 // 4, hidden_dim * 2, )),  # Reshape to 3D
            keras.layers.Conv1DTranspose(
                hidden_dim * 2, kernel_size=3, strides=2, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            # First residual block in decoder
            ResidualBlock(hidden_dim * 2, kernel_size=3),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1DTranspose(
                hidden_dim, kernel_size=3, strides=2, activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            # Second residual block in decoder
            ResidualBlock(hidden_dim, kernel_size=5),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1DTranspose(1, kernel_size=7, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Reshape((128,)),
            keras.layers.Dense(input_dim, activation='relu'),
            keras.layers.Dense(input_dim)
        ])

# @tf.keras.utils.register_keras_serializable()
# class ConvVAE(VAE):
#     def __init__(self, input_dim, latent_dim, hidden_dim, **kwargs):
#         super(ConvVAE, self).__init__(
#             input_dim, latent_dim, hidden_dim, **kwargs)

#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.hidden_dim = hidden_dim

#         # Encoder with residual blocks
#         self.encoder = self.build_encoder_with_skip_connections()

#         # Latent space layers
#         self.z_mean = keras.layers.Dense(latent_dim)
#         self.z_log_var = keras.layers.Dense(latent_dim)
#         self.sampling = Sampling()

#         # Decoder with residual blocks
#         self.decoder = keras.Sequential([
#             keras.layers.InputLayer(input_shape=(latent_dim,)),
#             keras.layers.Dense(
#                 hidden_dim * 2 * (128 // 4), activation="relu"),
#             keras.layers.Reshape(
#                 (128 // 4, hidden_dim * 2, )),  # Reshape to 3D
#             keras.layers.Conv1DTranspose(
#                 hidden_dim * 2, kernel_size=3, strides=2, activation="relu", padding="same"),
#             # First residual block in decoder
#             ResidualBlock(hidden_dim * 2, kernel_size=3),
#             keras.layers.Conv1DTranspose(
#                 hidden_dim, kernel_size=3, strides=2, activation="relu", padding="same"),
#             # Second residual block in decoder
#             ResidualBlock(hidden_dim, kernel_size=5),
#             keras.layers.Conv1DTranspose(1, kernel_size=7, padding="same"),
#             keras.layers.Reshape((128,)),
#             keras.layers.Dense(input_dim, activation='relu'),
#             keras.layers.Dense(input_dim)
#         ])

#     def build_encoder_with_skip_connections(self, activation_fn='relu'):
#         """
#         Builds a 1D CNN encoder with skip connections for input data of specified length.

#         Parameters:
#             activation_fn (str): The activation function to use in the layers.

#         Returns:
#             tf.keras.Model: The encoder model.
#         """
#         inputs = keras.layers.Input(
#             shape=(self.input_dim, 1))  # Input shape (126, 1)

#         # First convolutional block
#         x1 = keras.layers.Conv1D(
#             128, kernel_size=3, strides=1, padding="same", activation=activation_fn)(inputs)
#         x1 = keras.layers.BatchNormalization()(x1)

#         x2 = keras.layers.Conv1D(128, kernel_size=3, strides=1,
#                                  padding="same", dilation_rate=1, activation=activation_fn)(x1)
#         x2 = keras.layers.BatchNormalization()(x2)

#         x3 = keras.layers.Conv1D(128, kernel_size=3, strides=1,
#                                  padding="same", dilation_rate=1, activation=activation_fn)(x2)
#         x3 = keras.layers.BatchNormalization()(x3)

#         # x4 = keras.layers.Concatenate()([x4, x1])

#         x5 = keras.layers.Conv1D(256, kernel_size=3, strides=2,
#                                  padding="same", activation=activation_fn)(x3)

#         x5 = keras.layers.BatchNormalization()(x5)
#         # Fourth convolutional block
#         x6 = keras.layers.Conv1D(256, kernel_size=3, strides=2,
#                                  padding="same", activation=activation_fn)(x5)

#         x6 = keras.layers.BatchNormalization()(x6)
#         x7 = keras.layers.Conv1D(256, kernel_size=3, strides=2,
#                                  padding="same", activation=activation_fn)(x6)
#         x7 = keras.layers.BatchNormalization()(x7)
#         outputs = keras.layers.Flatten()(x7)

#         return keras.Model(inputs, outputs, name="Encoder_with_Skip_Connections")


def vae_loss(inputs, reconstructed, z_mean, z_log_var):
    mse_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
    kl_loss = -0.5 * \
        tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    total_loss = mse_loss + 0.01 * kl_loss
    return total_loss


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var = model(x)
        loss = vae_loss(x, reconstructed, z_mean, z_log_var)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_vae(model, train_dataset, optimizer, epochs):
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for step, x in enumerate(train_dataset):
            loss = train_step(model, x, optimizer)
            epoch_loss.update_state(loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss.result():.4f}")

        if (epoch > 0) and (epoch % 50 == 0):
            model.save(
                f"../../models/vae/ConvBased/checkpoints/LOC1-LOC2-e{epoch + 400}-mse1-kl0.01-Conv-ldim{latent_dim}-hdim{hidden_dim}.keras")


def filter_and_sort_data(df, location_label):
    """
    Filter the dataframe by Location, sort by Website, and drop specified columns.
    """
    return df[df['Location'] == location_label].sort_values(by=['Website']).iloc[:, 2:].reset_index(drop=True)


if __name__ == '__main__':

    import init_gpu
    import init_dataset
    n_neurons = 32

    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # location = sys.argv[1]
    # if location not in locations:
    #     raise sys.exit('Invalid Location, Exiting...')

    # x_train = filter_and_sort_data(
    #     train_df, location).to_numpy().astype(np.float32)
    # both platforms
    x_train = train_df.iloc[:, 2:].to_numpy().astype(np.float32)

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        x_train).shuffle(buffer_size=10000).batch(128)

    # Initialize model
    input_dim = length  # 126
    latent_dim = 96
    hidden_dim = 128
    # latent_dim = 64
    # hidden_dim = 96
    model = ConvVAE(input_dim, latent_dim, hidden_dim)
    print(model.encoder.summary())
    print(model.decoder.summary())
    model: ConvVAE = tf.keras.models.load_model(
        f"../../models/vae/ConvBased/LOC1-LOC2-e400-mse1-kl0.01-Conv-ldim96-hdim128.keras", custom_objects={'ConvVAE': ConvVAE, 'Sampling': Sampling})

    model.encoder.trainable = False
    model.z_mean.trainable = False
    model.z_log_var.trainable = False
    model.decoder.trainable = True

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Train the model
    epochs = 400
    train_vae(model, train_dataset, optimizer, epochs=epochs)

    # Save the model
    model.save(
        f"../../models/vae/ConvBased/LOC1-LOC2-e{epochs + 800}-mse1-kl0.01-Conv-ldim{latent_dim}-hdim{hidden_dim}.keras")
