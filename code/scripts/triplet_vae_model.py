import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@tf.keras.utils.register_keras_serializable()
class VAE_Triplet_Model(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_domains, **kwargs):
        super(VAE_Triplet_Model, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains

        # Ensure all layers use float32
        tf.keras.backend.set_floatx('float32')

        # VAE Encoder
        self.vae_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Reshape((1, input_dim, )),
            tf.keras.layers.GRU(
                hidden_dim, return_sequences=True, recurrent_dropout=0.1),
            tf.keras.layers.GRU(
                hidden_dim, return_sequences=False, recurrent_dropout=0.1),
        ])
        self.z_mean = tf.keras.layers.Dense(latent_dim)
        self.z_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

        # Conditional Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=(latent_dim + num_domains,)),
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.RepeatVector(1),  # Prepare for GRU layers
            tf.keras.layers.GRU(
                hidden_dim, return_sequences=True, activation="relu"),
            tf.keras.layers.GRU(
                input_dim, return_sequences=False, activation="relu"),
            tf.keras.layers.Dense(
                input_dim, activation='linear')  # Output layer
        ])

    def encode(self, x):
        x = self.vae_encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    @tf.function
    def decode(self, z, domain):
        z_with_domain = tf.concat([z, domain], axis=-1)
        return self.decoder(z_with_domain)

    def call(self, inputs):
        anchor, positive, negative, domain = inputs

        # VAE encoding first
        anchor_mean, anchor_log_var, anchor_z = self.encode(anchor)
        positive_mean, positive_log_var, positive_z = self.encode(positive)
        negative_mean, negative_log_var, negative_z = self.encode(negative)

        # Decoding using sampled z
        reconstructed = self.decode(anchor_z, domain)

        # Return means for triplet loss calculation
        return reconstructed, anchor_mean, anchor_log_var, anchor_mean, positive_mean, negative_mean

    def get_config(self):
        config = super(VAE_Triplet_Model,  self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_domains": self.num_domains,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class Pretrained_Triplet_VAE(VAE_Triplet_Model):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_platforms, **kwargs):
        super(Pretrained_Triplet_VAE, self).__init__(
            input_dim, latent_dim, hidden_dim, num_platforms, **kwargs)

        self.vae_encoder = tf.keras.models.load_model(
            "/home/asil0892/doh_traffic_analysis/models/website/LOC1-LOC2-baseGRU-epochs200-train_samples1200-triplet_samples5.keras")
        self.vae_encoder.trainable = True


@tf.keras.utils.register_keras_serializable()
class VAE_Triplet_Model_V2(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_domains, **kwargs):
        super(VAE_Triplet_Model_V2, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        # Ensure all layers use float32
        tf.keras.backend.set_floatx('float32')

        # VAE Encoder
        self.vae_encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Reshape((1, input_dim,)),
            # tf.keras.layers.GRU(
            #     hidden_dim, return_sequences=True, recurrent_dropout=0.1),
            tf.keras.layers.LSTM(
                hidden_dim, return_sequences=False, recurrent_dropout=0.1),
        ])
        self.z_mean = tf.keras.layers.Dense(latent_dim)
        self.z_log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

        # Conditional Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=(latent_dim + num_domains,)),
            tf.keras.layers.Dense(hidden_dim, activation="tanh"),
            tf.keras.layers.RepeatVector(1),
            # tf.keras.layers.LSTM(
            #     hidden_dim, return_sequences=True, activation="tanh"),
            tf.keras.layers.LSTM(
                input_dim, return_sequences=False, activation="tanh"),
            tf.keras.layers.Dense(
                input_dim, activation='linear')
        ])

    def encode(self, x):
        x = self.vae_encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

    @tf.function
    def decode(self, z, domain):
        z_with_domain = tf.concat([z, domain], axis=-1)
        return self.decoder(z_with_domain)

    def call(self, inputs):
        anchor, positive, negative, anchor_domain, positive_domain = inputs

        # VAE encoding
        anchor_mean, anchor_log_var, anchor_z = self.encode(anchor)
        positive_mean, positive_log_var, positive_z = self.encode(positive)
        negative_mean, negative_log_var, negative_z = self.encode(negative)

        # Four reconstructions
        # 1. anchor content with anchor domain
        recon_anchor_anchor = self.decode(anchor_z, anchor_domain)
        # 2. anchor content with positive domain
        recon_anchor_positive = self.decode(anchor_z, positive_domain)
        # 3. positive content with anchor domain
        recon_positive_anchor = self.decode(positive_z, anchor_domain)
        # 4. positive content with positive domain
        recon_positive_positive = self.decode(positive_z, positive_domain)

        # Return all necessary components for loss calculation
        return {
            'recon_anchor_anchor': recon_anchor_anchor,
            'recon_anchor_positive': recon_anchor_positive,
            'recon_positive_anchor': recon_positive_anchor,
            'recon_positive_positive': recon_positive_positive,
            'anchor_mean': anchor_mean,
            'anchor_log_var': anchor_log_var,
            'positive_mean': positive_mean,
            'positive_log_var': positive_log_var,
            'negative_mean': negative_mean,
            'negative_log_var': negative_log_var,
        }

    def get_config(self):
        config = super(VAE_Triplet_Model_V2, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_domains": self.num_domains,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
