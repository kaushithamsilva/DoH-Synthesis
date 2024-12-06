import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sys


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


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
class ConditionalVAE(keras.Model):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dim, **kwargs):
        super(ConditionalVAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(
                input_dim + condition_dim, 1)),
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

        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim + condition_dim,)),
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

    def encode(self, x, y):
        # Concatenate x and condition y
        x_cond = tf.concat([x, y], axis=-1)
        x = self.encoder(x_cond)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    def decode(self, z, y):
        # Concatenate z and condition y
        z_cond = tf.concat([z, y], axis=-1)
        return self.decoder(z_cond)

    def call(self, inputs):
        x, y = inputs
        z_mean, z_log_var, z = self.encode(x, y)
        reconstructed = self.decode(z, y)
        return reconstructed, z_mean, z_log_var

    def get_config(self):
        config = super(ConditionalVAE, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "condition_dim": self.condition_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def vae_loss(inputs, reconstructed, z_mean, z_log_var):
    mse_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
    kl_loss = -0.5 * \
        tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    total_loss = mse_loss + 0.01 * kl_loss
    return total_loss

# Update train step to pass (X, y) as input


@tf.function
def train_step(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var = model((x, y))
        loss = vae_loss(x, reconstructed, z_mean, z_log_var)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Update train function


def train_vae(model, train_dataset, optimizer, epochs):
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        for step, (x, y) in enumerate(train_dataset):
            loss = train_step(model, x, y, optimizer)
            epoch_loss.update_state(loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss.result():.4f}")

        if (epoch > 0) and (epoch % 5 == 0):
            print("Checkpoint: Saving Model...")
            model.save(
                f"../../models/vae/conditional_vae/checkpoints/LOC1-LOC2-e{epoch}-mse1-kl0.01.keras")


def filter_and_sort_data(df, location_label):
    """
    Filter the dataframe by Location, sort by Website, and drop specified columns.
    """
    return df[df['Location'] == location_label].sort_values(by=['Website']).iloc[:, 2:].reset_index(drop=True)


def get_web_embeddings(data, web_model):
    embeddings = []
    chunk_size = 10000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        transformed_chunk = web_model(chunk)
        embeddings.append(transformed_chunk)

    return np.vstack(embeddings)


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

    web_model = tf.keras.models.load_model(
        "../../models/website/LOC1-LOC2-baseGRU-epochs100-train_samples1200-triplet_samples5-domain_invariant-l1.keras")
    length = len(df.columns) - 2  # subtract the two label columns

    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    one_hot_encoded_loc = {'LOC1': np.array([0.0, 1.0]).astype(
        'float32'), 'LOC2': np.array([1.0, 0.0]).astype('float32')}

    # prepare tensorflow dataset
    # for location 1 get all the data available
    x_LOC1 = filter_and_sort_data(
        df, 'LOC1').to_numpy().astype(np.float32)

    print(f"x_LOC1 shape: {x_LOC1.shape}")

    # neglect the test set for the the location 2 data.
    # these are the missing classes which will be used to test the model
    x_LOC2 = filter_and_sort_data(
        train_df, 'LOC2').to_numpy().astype(np.float32)

    x_LOC1_web_embedding = get_web_embeddings(
        x_LOC1, web_model)
    x_LOC2_web_embedding = get_web_embeddings(
        x_LOC2, web_model)

    LOC1_domain_labels = np.tile(one_hot_encoded_loc['LOC1'], (len(x_LOC1), 1))
    LOC2_domain_labels = np.tile(one_hot_encoded_loc['LOC2'], (len(x_LOC2), 1))

    # input to the VAE: (trace: x, condition: {web(x), domain(x)})
    y_LOC1 = np.hstack((x_LOC1_web_embedding, LOC1_domain_labels))
    y_LOC2 = np.hstack((x_LOC2_web_embedding, LOC2_domain_labels))

    # X is the trace, y is the condition
    X = np.vstack((x_LOC1, x_LOC2))
    y = np.vstack((y_LOC1, y_LOC2))

    # Debugging Shapes
    print(f"X shape: {X.shape}, Condition shape: {y.shape}")

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X, y)).shuffle(buffer_size=10000).batch(32)

    # Initialize model
    input_dim = length  # 126
    latent_dim = 96
    condition_dim = y.shape[1]
    hidden_dim = 128
    model = ConditionalVAE(input_dim, condition_dim, latent_dim, hidden_dim)

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Train the model
    epochs = 400
    train_vae(model, train_dataset, optimizer, epochs=epochs)

    # Save the model
    model.save(
        f"../../models/vae/conditional_vae/LOC1-LOC2-e{epochs}-mse1-kl0.01.keras")
