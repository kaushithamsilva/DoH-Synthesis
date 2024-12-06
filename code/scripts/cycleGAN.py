import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd


def ResidualBlock(filters, kernel_size):
    """
    Defines a residual block with Conv1D, BatchNormalization, and skip connections.
    Includes a 1x1 convolution on the shortcut path if needed.
    """
    def block(x):
        shortcut = x  # Save the input for the skip connection

        # First convolutional layer
        x = layers.Conv1D(filters, kernel_size,
                          padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # Second convolutional layer
        x = layers.Conv1D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Adjust the shortcut to match the transformed input's shape
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(
                filters, kernel_size=1, padding="same")(shortcut)

        return layers.add([shortcut, x])  # Add the shortcut connection

    return block

# Define the generator model


def build_generator(input_dim, hidden_dim, output_dim):
    """
    Generator model inspired by the VAE encoder without a bottleneck.
    Transforms data from one domain to another.
    """
    inputs = tf.keras.Input(shape=(input_dim, 1))

    # Initial Convolution
    x = layers.Conv1D(hidden_dim, kernel_size=7, strides=1,
                      activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    # Residual Block 1
    x = ResidualBlock(hidden_dim, kernel_size=5)(x)

    # Residual Block 2
    x = ResidualBlock(hidden_dim, kernel_size=5)(x)

    # Residual Block 3
    x = ResidualBlock(hidden_dim, kernel_size=5)(x)

    # Final Convolution to generate the output
    outputs = layers.Conv1D(1, kernel_size=7, strides=1,
                            activation="tanh", padding="same")(x)

    return tf.keras.Model(inputs, outputs, name="Generator")


# Define the discriminator model
def build_discriminator(input_dim, hidden_dim):
    """
    Discriminator using dense and convolutional layers with residual blocks for moderate complexity.
    """
    inputs = tf.keras.Input(
        shape=(input_dim, 1))  # Assuming input reshaped to (input_dim, 1)

    # Initial Dense Layer
    x = layers.Conv1D(hidden_dim, kernel_size=7, strides=2,
                      activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)

    # Residual Block 1
    x = ResidualBlock(hidden_dim, kernel_size=5)(x)

    # Residual Block 2
    x = ResidualBlock(hidden_dim * 2, kernel_size=5)(x)

    # Flatten and Fully Connected Layers
    x = layers.Flatten()(x)
    x = layers.Dense(hidden_dim, activation="relu")(x)
    x = layers.Dense(hidden_dim // 2, activation="relu")(x)
    outputs = layers.Dense(1)(x)  # No activation for WGAN

    return tf.keras.Model(inputs, outputs, name="Discriminator")


# Wasserstein Gradient Penalty loss
def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def gradient_penalty(discriminator, real_samples, fake_samples, batch_size, LAMBDA=10.0):
    """
    Computes the gradient penalty for WGAN.
    """
    epsilon = tf.random.normal([batch_size, 1, 1], mean=0.0, stddev=1.0)
    interpolated_samples = epsilon * \
        real_samples + (1 - epsilon) * fake_samples
    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_output = discriminator(
            interpolated_samples, training=True)
    gradients = tape.gradient(interpolated_output, interpolated_samples)
    gradients = tf.reshape(gradients, [batch_size, -1])
    gradient_norm = tf.norm(gradients, axis=1)
    return LAMBDA * tf.reduce_mean(tf.square(gradient_norm - 1.0))


# Define CycleGAN model
class CycleGAN(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(CycleGAN, self).__init__()
        # Generators
        self.gen_g = build_generator(
            input_dim, hidden_dim, input_dim)  # G: LOC1 -> LOC2
        self.gen_f = build_generator(
            input_dim, hidden_dim, input_dim)  # F: LOC2 -> LOC1

        # Discriminators
        self.disc_x = build_discriminator(input_dim, hidden_dim)  # D_X: LOC1
        self.disc_y = build_discriminator(input_dim, hidden_dim)  # D_Y: LOC2

    def compile(self, gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer, cycle_loss_fn):
        super(CycleGAN, self).compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.cycle_loss_fn = cycle_loss_fn

    def train_step(self, data):
        real_x, real_y = data  # real_x: LOC1 samples, real_y: LOC2 samples
        batch_size = tf.shape(real_x)[0]

        with tf.GradientTape(persistent=True) as tape:
            # Forward cycle: LOC1 -> LOC2 -> LOC1
            fake_y = self.gen_g(real_x, training=True)
            cycled_x = self.gen_f(fake_y, training=True)

            # Backward cycle: LOC2 -> LOC1 -> LOC2
            fake_x = self.gen_f(real_y, training=True)
            cycled_y = self.gen_g(fake_x, training=True)

            # Discriminator outputs
            disc_real_x = self.disc_x(real_x, training=True)
            disc_fake_x = self.disc_x(fake_x, training=True)

            disc_real_y = self.disc_y(real_y, training=True)
            disc_fake_y = self.disc_y(fake_y, training=True)

            # Generator losses (Wasserstein)
            gen_g_loss = wasserstein_loss(
                tf.ones_like(disc_fake_y), disc_fake_y)
            gen_f_loss = wasserstein_loss(
                tf.ones_like(disc_fake_x), disc_fake_x)

            # Cycle-consistency losses
            cycle_loss_x = self.cycle_loss_fn(real_x, cycled_x)
            cycle_loss_y = self.cycle_loss_fn(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + cycle_loss_x
            total_gen_f_loss = gen_f_loss + cycle_loss_y

            # Discriminator losses (Wasserstein)
            disc_x_loss = wasserstein_loss(tf.ones_like(disc_real_x), disc_real_x) - \
                wasserstein_loss(tf.ones_like(disc_fake_x), disc_fake_x)
            disc_y_loss = wasserstein_loss(tf.ones_like(disc_real_y), disc_real_y) - \
                wasserstein_loss(tf.ones_like(disc_fake_y), disc_fake_y)

            # Add gradient penalty to discriminator losses
            disc_x_loss += gradient_penalty(self.disc_x,
                                            real_x, fake_x, batch_size)
            disc_y_loss += gradient_penalty(self.disc_y,
                                            real_y, fake_y, batch_size)

        # Calculate gradients
        gen_g_gradients = tape.gradient(
            total_gen_g_loss, self.gen_g.trainable_variables)
        gen_f_gradients = tape.gradient(
            total_gen_f_loss, self.gen_f.trainable_variables)
        disc_x_gradients = tape.gradient(
            disc_x_loss, self.disc_x.trainable_variables)
        disc_y_gradients = tape.gradient(
            disc_y_loss, self.disc_y.trainable_variables)

        # Apply gradients
        self.gen_g_optimizer.apply_gradients(
            zip(gen_g_gradients, self.gen_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(
            zip(gen_f_gradients, self.gen_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(
            zip(disc_x_gradients, self.disc_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(
            zip(disc_y_gradients, self.disc_y.trainable_variables))

        return {
            "gen_g_loss": total_gen_g_loss,
            "gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss,
        }


def filter_and_sort_data(df, location_label):
    """
    Filter the dataframe by Location, sort by Website, and drop specified columns.
    """
    return df[df['Location'] == location_label].sort_values(by=['Website']).iloc[:, 2:].reset_index(drop=True)


if __name__ == '__main__':

    import init_gpu
    import init_dataset
    init_gpu.initialize_gpus()

    locations = ['LOC2', 'LOC3']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # Hyperparameters
    input_dim = length  # Number of features (126 in your case)
    latent_dim = 96
    hidden_dim = 128
    learning_rate = 0.001
    lambda_cycle = 10.0

    # Ensure float32 type and reshape
    X_train_source_location = tf.data.Dataset.from_tensor_slices(
        filter_and_sort_data(train_df, locations[0])
        .to_numpy().astype(np.float32)
        .reshape(-1, input_dim, 1)
    )

    X_train_target_location = tf.data.Dataset.from_tensor_slices(
        filter_and_sort_data(train_df, locations[1])
        .to_numpy().astype(np.float32)
        .reshape(-1, input_dim, 1)
    )

    # Batch the datasets
    X_train_source_location = X_train_source_location.batch(128)
    X_train_target_location = X_train_target_location.batch(128)

    # Zip the datasets
    combined_dataset = tf.data.Dataset.zip(
        (X_train_source_location, X_train_target_location))

    # Initialize model
    cyclegan = CycleGAN(input_dim, latent_dim, hidden_dim)
    cyclegan.compile(
        gen_g_optimizer=tf.keras.optimizers.Adam(learning_rate),
        gen_f_optimizer=tf.keras.optimizers.Adam(learning_rate),
        disc_x_optimizer=tf.keras.optimizers.Adam(learning_rate),
        disc_y_optimizer=tf.keras.optimizers.Adam(learning_rate),
        cycle_loss_fn=lambda x, y: tf.reduce_mean(
            tf.square(x - y)) * lambda_cycle,
    )

    # Training
    cyclegan.fit(
        combined_dataset,
        epochs=50,
        shuffle=True,
    )

    # After training
    print("Training complete. Saving the generator model for LOC1 -> LOC2 translation...")

    # Save the generator that maps from LOC1 to LOC2 (source to target translation)
    cyclegan.gen_g.save(
        "../../models-LOC2-LOC3/CycleGAN/generator_LOC2_to_LOC3.keras")
    cyclegan.gen_g.save_weights(
        "../../models-LOC2-LOC3/CycleGAN/generator_LOC2_to_LOC3.h5")
