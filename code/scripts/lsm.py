"""
Latent Space Mapping: Bridging VAE for Cross Domain Alignment
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class SharedEncoder(keras.layers.Layer):
    def __init__(self, hidden_dim, latent_dim, num_domains=2, name="shared_encoder"):
        super(SharedEncoder, self).__init__(name=name)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_domains = num_domains

        # Increase network capacity with larger hidden layers
        self.encoder_net = keras.Sequential([
            keras.layers.Dense(hidden_dim * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
        ])

        self.mu = keras.layers.Dense(latent_dim)
        self.log_var = keras.layers.Dense(latent_dim)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=None):
        x, domain_label = inputs  # Unpack the inputs

        # Concatenate input and domain label
        concat = tf.concat([x, domain_label], axis=1)

        # Pass through encoder network
        x = self.encoder_net(concat, training=training)

        # Get latent parameters
        mu = self.mu(x)
        log_var = self.log_var(x)

        # Reparameterization trick
        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + tf.exp(0.5 * log_var) * eps

        return z, mu, log_var


class SharedDecoder(keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim, num_domains=2, name="shared_decoder"):
        super(SharedDecoder, self).__init__(name=name)

        self.decoder_net = keras.Sequential([
            keras.layers.Dense(hidden_dim * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_dim)
        ])

    def call(self, inputs, training=None):
        z, domain_label = inputs  # Unpack inputs
        concat = tf.concat([z, domain_label], axis=1)
        return self.decoder_net(concat, training=training)


class Classifier(keras.layers.Layer):
    def __init__(self, hidden_dim, num_classes, name="classifier"):
        super(Classifier, self).__init__(name=name)

        # Hierarchical classification network for better handling of many classes
        self.feature_extractor = keras.Sequential([
            keras.layers.Dense(hidden_dim * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim * 2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.BatchNormalization(),
        ])

        # Final classification layer with sparse categorical crossentropy
        self.classifier_head = keras.layers.Dense(num_classes)

    def call(self, z, training=False):
        features = self.feature_extractor(z, training=training)
        logits = self.classifier_head(features)
        return logits


class BridgingVAE(keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes, num_domains=2):
        super(BridgingVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = SharedEncoder(hidden_dim, latent_dim, num_domains)
        self.decoder = SharedDecoder(hidden_dim, input_dim, num_domains)
        self.classifier = Classifier(hidden_dim, num_classes)

    def encode(self, x, domain_label, training=None):
        return self.encoder((x, domain_label), training=training)

    def decode(self, z, domain_label, training=None):
        return self.decoder((z, domain_label), training=training)

    def classify(self, z, training=None):
        return self.classifier(z, training=training)

    def compute_swd_loss(self, z_A, z_B, num_projections=50):
        projections = tf.random.normal([num_projections, self.latent_dim])
        projections = projections / tf.norm(projections, axis=1, keepdims=True)

        proj_A = tf.matmul(z_A, projections, transpose_b=True)
        proj_B = tf.matmul(z_B, projections, transpose_b=True)

        proj_A_sorted = tf.sort(proj_A, axis=0)
        proj_B_sorted = tf.sort(proj_B, axis=0)

        return tf.reduce_mean(tf.square(proj_A_sorted - proj_B_sorted))

    @tf.function
    def train_step(self, data):
        x_A, x_B, x_A_domain, x_B_domain, y_A, y_B = data

        with tf.GradientTape() as tape:
            # Encode
            z_A, mu_A, log_var_A = self.encode(x_A, x_A_domain, training=True)
            z_B, mu_B, log_var_B = self.encode(x_B, x_B_domain, training=True)

            # Decode
            x_A_recon = self.decode(z_A, x_A_domain, training=True)
            x_B_recon = self.decode(z_B, x_B_domain, training=True)

            # Classify
            logits_A = self.classify(z_A, training=True)
            logits_B = self.classify(z_B, training=True)
            # Check for NaN in logits
            tf.debugging.check_numerics(
                logits_A, "Logits_A contains NaN or Inf")
            tf.debugging.check_numerics(
                logits_B, "Logits_B contains NaN or Inf")
            # Compute losses
            recon_loss_A = tf.reduce_mean(tf.square(x_A - x_A_recon))
            recon_loss_B = tf.reduce_mean(tf.square(x_B - x_B_recon))
            recon_loss = recon_loss_A + recon_loss_B

            kl_loss_A = -0.5 * \
                tf.reduce_mean(1 + log_var_A -
                               tf.square(mu_A) - tf.exp(log_var_A))
            kl_loss_B = -0.5 * \
                tf.reduce_mean(1 + log_var_B -
                               tf.square(mu_B) - tf.exp(log_var_B))
            kl_loss = kl_loss_A + kl_loss_B

            swd_loss = self.compute_swd_loss(z_A, z_B)

            cls_loss_A = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_A, logits_A, from_logits=True
                )
            )
            cls_loss_B = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_B, logits_B, from_logits=True
                )
            )
            cls_loss = cls_loss_A + cls_loss_B

            total_loss = (
                1.0 * recon_loss +
                0.01 * kl_loss +
                0.01 * swd_loss +
                1.0 * cls_loss
            )

        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "swd_loss": swd_loss,
            "cls_loss": cls_loss
        }

    def call(self, inputs):
        x, domain_label = inputs
        z, _, _ = self.encode(x, domain_label)
        return self.decode(z, domain_label)


if __name__ == '__main__':

    import init_gpu
    import init_dataset
    from train_vae import VAE, Sampling

    def data_preprocessing(df, location_label, vae: VAE):
        """
        Filter the dataframe by Location, sort by Website, and drop specified columns.
        """

        le = LabelEncoder()
        loc_df = df[df['Location'] == location_label].sort_values(by=[
                                                                  'Website'])
        loc_data = loc_df.iloc[:, 2:].to_numpy()
        loc_websites = le.fit_transform(loc_df.Website.to_numpy())

        _, _, loc_latent_embeddings = vae.encode(loc_data)
        return loc_latent_embeddings, loc_websites

    n_neurons = 32

    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2']
    one_hot_encoded_loc = {'LOC1': np.array([0.0, 1.0]).astype(
        'float32'), 'LOC2': np.array([1.0, 0.0]).astype('float32')}

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # Load VAE models
    LOC1_vae = tf.keras.models.load_model(
        "../../models/vae/LOC1-e400-mse1-kl0.01.keras")
    LOC2_vae = tf.keras.models.load_model(
        "../../models/vae/LOC2-e400-mse1-kl0.01.keras")

    x_LOC1, y_LOC1 = data_preprocessing(train_df, 'LOC1', LOC1_vae)
    x_LOC2, y_LOC2 = data_preprocessing(train_df, 'LOC2', LOC2_vae)

    # Convert labels to one-hot encoding
    # y_LOC1_onehot = tf.keras.utils.to_categorical(y_LOC1)
    # y_LOC2_onehot = tf.keras.utils.to_categorical(y_LOC2)

    # Create domain labels
    LOC1_domain_labels = np.tile(one_hot_encoded_loc['LOC1'], (len(x_LOC1), 1))
    LOC2_domain_labels = np.tile(one_hot_encoded_loc['LOC2'], (len(x_LOC2), 1))

    # Model parameters
    input_dim = n_neurons
    hidden_dim = 128
    latent_dim = n_neurons
    num_classes = 1200
    batch_size = 128
    epochs = 1000

    # Initialize the BridgingVAE model
    model = BridgingVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_classes=num_classes
    )

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    # Create tf.data.Dataset for training
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {
            'x_A': x_LOC1,
            'x_B': x_LOC2,
            'domain_A': LOC1_domain_labels,
            'domain_B': LOC2_domain_labels,
            'y_A': y_LOC1,
            'y_B': y_LOC2
        }
    )).shuffle(buffer_size=10000).batch(batch_size)

    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_swd_loss = 0
        total_cls_loss = 0

        for batch in train_dataset:
            batch_data = (
                batch['x_A'], batch['x_B'],
                batch['domain_A'], batch['domain_B'],
                batch['y_A'], batch['y_B']
            )

            losses = model.train_step(batch_data)
            # print(losses)
            total_loss += losses['total_loss']
            total_recon_loss += losses['recon_loss']
            total_kl_loss += losses['kl_loss']
            total_swd_loss += losses['swd_loss']
            total_cls_loss += losses['cls_loss']
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_swd_loss = total_swd_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        print(
            f"\tRecon:{avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, SWD: {avg_swd_loss:.4f}, CLS: {avg_cls_loss:.4f}")

    print("Training completed!")

    # Save the model
    model.save_weights(
        '../../models/vae/bridging_vae_weights-1-0.01-0.01-1.weights.h5')
