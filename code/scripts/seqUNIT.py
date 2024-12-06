import tensorflow as tf
from tensorflow import keras
import numpy as np


class SharedEncoder(keras.Model):
    def __init__(self, hidden_dims, latent_dim, **kwargs):
        super(SharedEncoder, self).__init__(**kwargs)

        self.encoder = keras.Sequential([
            keras.layers.Dense(hidden_dims[0]),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),

            keras.layers.Dense(hidden_dims[1]),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),

            keras.layers.Reshape((1, hidden_dims[1])),
            keras.layers.GRU(
                hidden_dims[2], return_sequences=True, recurrent_dropout=0.1),
            keras.layers.GRU(hidden_dims[3], recurrent_dropout=0.1),
        ])

        # VAE components
        self.mu = keras.layers.Dense(latent_dim)
        self.log_var = keras.layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.encoder(inputs)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def encode(self, inputs):
        mu, log_var = self(inputs)
        z = self.reparameterize(mu, log_var)
        return z

    def reparameterize(self, mu, log_var):
        std = tf.exp(0.5 * log_var)
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + eps * std


class SequenceDecoder(keras.Model):
    def __init__(self, hidden_dims, output_dim, **kwargs):
        super(SequenceDecoder, self).__init__(**kwargs)

        self.decoder = keras.Sequential([
            keras.layers.Dense(hidden_dims[3]),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(),

            keras.layers.Reshape((1, hidden_dims[3])),
            keras.layers.GRU(
                hidden_dims[2], return_sequences=True, recurrent_dropout=0.1),
            keras.layers.GRU(
                hidden_dims[1], return_sequences=False, recurrent_dropout=0.1),

            keras.layers.Dense(hidden_dims[0]),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(),

            keras.layers.Dense(output_dim),
        ])

    def call(self, inputs):
        return self.decoder(inputs)


class SequenceDiscriminator(keras.Model):
    def __init__(self, hidden_dims, **kwargs):
        super(SequenceDiscriminator, self).__init__(**kwargs)

        self.discriminator = keras.Sequential([
            keras.layers.Dense(hidden_dims[0]),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.1),

            keras.layers.Reshape((1, hidden_dims[0])),
            keras.layers.GRU(
                hidden_dims[1], return_sequences=True, recurrent_dropout=0.1),
            keras.layers.GRU(hidden_dims[2], recurrent_dropout=0.1),

            keras.layers.Dense(hidden_dims[3]),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Dropout(0.1),

            keras.layers.Dense(1)
        ])

    def call(self, inputs):
        return self.discriminator(inputs)


class SequenceUNIT(keras.Model):
    def __init__(self, trace_dim, hidden_dims=[128, 256, 128, 64], latent_dim=32, **kwargs):
        super(SequenceUNIT, self).__init__(**kwargs)

        # Initialize components
        self.shared_encoder = SharedEncoder(hidden_dims, latent_dim)
        self.decoder_A = SequenceDecoder(hidden_dims, trace_dim)
        self.decoder_B = SequenceDecoder(hidden_dims, trace_dim)
        self.discriminator_A = SequenceDiscriminator(hidden_dims)
        self.discriminator_B = SequenceDiscriminator(hidden_dims)

        self.latent_dim = latent_dim
        self.trace_dim = trace_dim

        # Initialize optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, beta_1=0.5)

    def encode(self, x, domain='A'):
        return self.shared_encoder.encode(x)

    def decode(self, z, domain='A'):
        if domain == 'A':
            return self.decoder_A(z)
        return self.decoder_B(z)

    def discriminate(self, x, domain='A'):
        if domain == 'A':
            return self.discriminator_A(x)
        return self.discriminator_B(x)

    @tf.function
    def train_step(self, batch_A, batch_B):
        # Get the discriminator and generator variables
        d_vars = (self.discriminator_A.trainable_variables +
                  self.discriminator_B.trainable_variables)
        g_vars = (self.shared_encoder.trainable_variables +
                  self.decoder_A.trainable_variables +
                  self.decoder_B.trainable_variables)

        # Training steps using gradient tape
        with tf.GradientTape(persistent=True) as tape:
            # Encode
            mu_A, log_var_A = self.shared_encoder(batch_A)
            mu_B, log_var_B = self.shared_encoder(batch_B)

            # Reparameterize
            z_A = self.shared_encoder.reparameterize(mu_A, log_var_A)
            z_B = self.shared_encoder.reparameterize(mu_B, log_var_B)

            # Cross-domain translations
            fake_B = self.decoder_B(z_A)
            fake_A = self.decoder_A(z_B)

            # Cycle reconstructions
            mu_fake_A, log_var_fake_A = self.shared_encoder(fake_A)
            mu_fake_B, log_var_fake_B = self.shared_encoder(fake_B)

            z_fake_A = self.shared_encoder.reparameterize(
                mu_fake_A, log_var_fake_A)
            z_fake_B = self.shared_encoder.reparameterize(
                mu_fake_B, log_var_fake_B)

            cycle_A = self.decoder_A(z_fake_A)
            cycle_B = self.decoder_B(z_fake_B)

            # Self reconstructions
            recon_A = self.decoder_A(z_A)
            recon_B = self.decoder_B(z_B)

            # Discriminator outputs
            d_real_A = self.discriminator_A(batch_A)
            d_fake_A = self.discriminator_A(fake_A)
            d_real_B = self.discriminator_B(batch_B)
            d_fake_B = self.discriminator_B(fake_B)

            # Calculate losses
            # VAE losses
            kl_loss_A = -0.5 * \
                tf.reduce_mean(1 + log_var_A -
                               tf.square(mu_A) - tf.exp(log_var_A))
            kl_loss_B = -0.5 * \
                tf.reduce_mean(1 + log_var_B -
                               tf.square(mu_B) - tf.exp(log_var_B))

            recon_loss_A = tf.reduce_mean(tf.square(batch_A - recon_A))
            recon_loss_B = tf.reduce_mean(tf.square(batch_B - recon_B))

            cycle_loss_A = tf.reduce_mean(tf.square(batch_A - cycle_A))
            cycle_loss_B = tf.reduce_mean(tf.square(batch_B - cycle_B))

            # GAN losses
            d_loss_A = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_A), logits=d_real_A) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(d_fake_A), logits=d_fake_A)
            )

            d_loss_B = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_B), logits=d_real_B) +
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(d_fake_B), logits=d_fake_B)
            )

            g_loss_A = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake_A), logits=d_fake_A)
            )
            g_loss_B = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake_B), logits=d_fake_B)
            )

            # Combined losses
            vae_loss = kl_loss_A + kl_loss_B + recon_loss_A + \
                recon_loss_B + cycle_loss_A + cycle_loss_B
            g_loss = g_loss_A + g_loss_B + vae_loss
            d_loss = d_loss_A + d_loss_B

        # Calculate gradients
        d_gradients = tape.gradient(d_loss, d_vars)
        g_gradients = tape.gradient(g_loss, g_vars)

        # Apply gradients
        self.d_optimizer.apply_gradients(zip(d_gradients, d_vars))
        self.g_optimizer.apply_gradients(zip(g_gradients, g_vars))

        return {
            'd_loss': d_loss,
            'g_loss': g_loss
        }

    def translate(self, x, source_domain='A', target_domain='B'):
        z = self.encode(x, source_domain)
        return self.decode(z, target_domain)


def train_model(model, train_dataset, epochs=500, save_interval=100):
    # Training metrics
    train_summary_writer = tf.summary.create_file_writer(
        '../../logs/unit_training')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Initialize metrics for this epoch
        epoch_d_losses = []
        epoch_g_losses = []

        # Progress bar for batches
        with tqdm(total=len(train_dataset)) as pbar:
            for batch_A, batch_B in train_dataset:
                # Training step
                losses = model.train_step(batch_A, batch_B)

                # Update metrics
                epoch_d_losses.append(losses['d_loss'].numpy())
                epoch_g_losses.append(losses['g_loss'].numpy())

                # Update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"D Loss: {losses['d_loss']:.4f}, G Loss: {losses['g_loss']:.4f}")

        # Calculate average losses for this epoch
        avg_d_loss = np.mean(epoch_d_losses)
        avg_g_loss = np.mean(epoch_g_losses)

        # Log metrics
        with train_summary_writer.as_default():
            tf.summary.scalar('discriminator_loss', avg_d_loss, step=epoch)
            tf.summary.scalar('generator_loss', avg_g_loss, step=epoch)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")

        # Save model periodically
        if (epoch + 1) % save_interval == 0:
            model_path = f"../../models/UNIT/unit_model_epoch_{epoch + 1}"
            model.save_weights(f"{model_path}/weights")


def create_tf_dataset(traces_loc1, traces_loc2, batch_size):
    """Create TensorFlow dataset maintaining paired samples"""
    # Create datasets
    dataset = tf.data.Dataset.from_tensor_slices((traces_loc1, traces_loc2))

    # Shuffle the paired data together
    dataset = dataset.shuffle(buffer_size=1000)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    return dataset


def prepare_training_data(df, locations):
    # Separate data by location
    loc1_data = df[df['Location'] == locations[0]].sort_values(by=[
        'Website'])
    loc2_data = df[df['Location'] == locations[1]].sort_values(by=[
        'Website'])

    # Extract traces (excluding Location and Website columns)
    traces_loc1 = loc1_data.iloc[:, 2:].values.astype(np.float32)
    traces_loc2 = loc2_data.iloc[:, 2:].values.astype(np.float32)

    return traces_loc1, traces_loc2


if __name__ == '__main__':
    import init_gpu as init_gpu
    import init_dataset as init_dataset
    import pandas as pd
    import tensorflow as tf
    import numpy as np
    from tqdm import tqdm

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

    # Prepare the data
    traces_loc1, traces_loc2 = prepare_training_data(train_df, locations)
    trace_dim = length  # dimension of the trace data

    # Create the UNIT model
    model = SequenceUNIT(
        trace_dim=trace_dim,
        hidden_dims=[128, 256, 128, 64],
        latent_dim=32
    )

    # Training parameters
    batch_size = 128
    epochs = 500
    save_interval = 100

    # Create dataset
    train_dataset = create_tf_dataset(traces_loc1, traces_loc2, batch_size)

    train_model(model, train_dataset, epochs, save_interval)
