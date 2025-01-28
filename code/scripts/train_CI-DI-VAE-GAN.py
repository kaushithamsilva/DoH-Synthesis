import tensorflow as tf
import numpy as np
import pandas as pd
"""
Use the VAE models and train a Class Informed - VAE (CI-VAE) to get the clusters of same class together as possible
"""


def vae_loss(inputs, reconstructed, z_mean, z_log_var):
    mse_loss = tf.reduce_mean(tf.square(inputs - reconstructed))
    kl_loss = -0.5 * \
        tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    total_loss = mse_loss + 0.1 * kl_loss
    return total_loss


def classification_loss(labels, predictions):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, predictions)

def gan_loss(real_outputs, fake_outputs):
    """GAN loss for discriminator and generator."""
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_outputs), real_outputs)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_outputs), fake_outputs)
    discriminator_loss = real_loss + fake_loss
    generator_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake_outputs), fake_outputs)
    return discriminator_loss, generator_loss


@tf.function
def train_step_ci_di_gan(vae_model, class_discriminator, domain_discriminator, gan_discriminator, x, y, d, optimizer, gan_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        # Forward pass through VAE
        reconstructed, z_mean, z_log_var = vae_model(x)
        vae_loss_value = vae_loss(x, reconstructed, z_mean, z_log_var)

        # Latent space classification losses
        z = z_mean
        class_preds = class_discriminator(z)
        class_loss_value = classification_loss(y, class_preds)
        domain_preds = domain_discriminator(z)
        domain_loss_value = classification_loss(d, domain_preds)

        # GAN discriminator loss
        real_outputs = gan_discriminator(x)
        fake_outputs = gan_discriminator(reconstructed)
        discriminator_loss_value, generator_loss_value = gan_loss(real_outputs, fake_outputs)

        # Combine losses
        total_vae_loss = vae_loss_value + class_loss_value + domain_loss_value + 0.1 * generator_loss_value
     # Compute gradients and update weights
    vae_grads = tape.gradient(
        total_vae_loss,
        vae_model.trainable_variables + class_discriminator.trainable_variables + domain_discriminator.trainable_variables
    )
    optimizer.apply_gradients(
        zip(vae_grads, vae_model.trainable_variables + class_discriminator.trainable_variables + domain_discriminator.trainable_variables)
    )

    # Update GAN discriminator separately
    discriminator_grads = tape.gradient(discriminator_loss_value, gan_discriminator.trainable_variables)
    gan_optimizer.apply_gradients(zip(discriminator_grads, gan_discriminator.trainable_variables))

    return total_vae_loss, vae_loss_value, class_loss_value, domain_loss_value, discriminator_loss_value, generator_loss_value


def train_ci_di_vae_gan(vae_model, class_discriminator, domain_discriminator, gan_discriminator, train_dataset, optimizer, gan_optimizer, epochs):
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_vae_loss = tf.keras.metrics.Mean()
        epoch_class_loss = tf.keras.metrics.Mean()
        epoch_domain_loss = tf.keras.metrics.Mean()
        epoch_discriminator_loss = tf.keras.metrics.Mean()
        epoch_generator_loss = tf.keras.metrics.Mean()

        for step, (x, y, d) in enumerate(train_dataset):
            total_loss, vae_loss_value, class_loss_value, domain_loss_value, discriminator_loss_value, generator_loss_value = train_step_ci_di_gan(
                vae_model, class_discriminator, domain_discriminator, gan_discriminator, x, y, d, optimizer, gan_optimizer
            )
            epoch_loss.update_state(total_loss)
            epoch_vae_loss.update_state(vae_loss_value)
            epoch_class_loss.update_state(class_loss_value)
            epoch_domain_loss.update_state(domain_loss_value)
            epoch_discriminator_loss.update_state(discriminator_loss_value)
            epoch_generator_loss.update_state(generator_loss_value)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss.result():.4f}, VAE Loss: {epoch_vae_loss.result():.4f}, "
              f"Class Loss: {epoch_class_loss.result():.4f}, Domain Loss: {epoch_domain_loss.result():.4f}, "
              f"Discriminator Loss: {epoch_discriminator_loss.result():.4f}, Generator Loss: {epoch_generator_loss.result():.4f}")

        if epoch % 20 == 0:
            vae_model.save(f"../../models-LOC2-LOC3/gan/ci_di_vae/ConvBased/domain_and_class/checkpoints/LOC2-LOC3-e{epoch}-vae.keras")
            gan_discriminator.save(f"../../models-LOC2-LOC3/gan/ci_di_vae/ConvBased/domain_and_class/checkpoints/LOC2-LOC3-e{epoch}-gan.keras")

def linear_discriminator(input_dim, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation=None,
                              input_shape=(input_dim,))
    ])

def gan_discriminator(input_dim):
    """GAN Discriminator network."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # Output: probability of "real"
    ])

if __name__ == '__main__':
    from train_vae import VAE, Sampling, filter_and_sort_data, ConvVAE, ConvVAE_BatchNorm
    from sklearn.preprocessing import LabelEncoder
    import init_gpu
    import init_dataset
    n_neurons = 32

    init_gpu.initialize_gpus()

    locations = ['LOC2', 'LOC3']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns
    print(length)
    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # for the ci vae use all the available data. only missing data from the target location's missing websites
    x_train = train_df.iloc[:, 2:].to_numpy().astype(np.float32)

    x_available_classes_from_source = test_df[test_df['Location']
                                              == locations[0]].iloc[:, 2:].to_numpy().astype(np.float32)

    x_train = np.vstack((x_train, x_available_classes_from_source))

    # get classes for class informed latent space
    le = LabelEncoder()
    y_train = train_df.Website
    y_available_classes = test_df[test_df['Location']
                                  == locations[0]].Website.to_numpy()
    y_train = le.fit_transform(np.hstack((y_train, y_available_classes)))

    # Get domain labels (0 for LOC1, 1 for LOC2)
    d_train = (train_df.Location == locations[1]).astype(int)
    d_available_classes = np.zeros(
        len(y_available_classes))  # All from LOC1, so 0
    d_train = np.hstack((d_train, d_available_classes))

    # check for any NaN or Infinite values
    assert not np.any(np.isnan(x_train)), "Input contains NaN values"
    assert not np.any(np.isinf(x_train)), "Input contains infinite values"
    assert set(d_train) == {0, 1}, "d_train contains values other than 0 and 1"
    print(
        f"y_train classes: {np.unique(y_train)}, d_train classes: {np.unique(d_train)}")

    # Create TensorFlow dataset with both class and domain labels
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, d_train)).shuffle(buffer_size=10000).batch(128)

    # Initialize vae_model
    input_dim = length  # 126
    latent_dim = 96
    hidden_dim = 128
    vae_model = ConvVAE_BatchNorm(input_dim, latent_dim, hidden_dim)

    class_discriminator = linear_discriminator(latent_dim, 1500)
    domain_discriminator = linear_discriminator(latent_dim, 2)
    gan_discriminator_model = gan_discriminator(input_dim)

    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    gan_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Train CI_DI_VAE_GAN
    epochs = 1000
    train_ci_di_vae_gan(vae_model, class_discriminator, domain_discriminator, gan_discriminator_model,
                        train_dataset, optimizer, gan_optimizer, epochs=epochs)

    # Save the vae_model
    vae_model.save(
        f"../../models-LOC2-LOC3/gan/ci_di_vae/ConvBased/domain_and_class/LOC2-LOC3-e{epochs}-mse1-kl_mixed-cl1.0-ConvBatchNorm-ldim{latent_dim}-hdim{hidden_dim}.keras")

    class_discriminator.save(
        f"../../models-LOC2-LOC3/gan/ci_di_vae/ConvBased/domain_and_class/linear_discriminators/class_discriminator-kl_mixed.keras")
    domain_discriminator.save(
        f"../../models-LOC2-LOC3/gan/ci_di_vae/ConvBased/domain_and_class/linear_discriminators/domain_discriminator-kl_mixed.keras")

    gan_discriminator_model.save("../../models-LOC2-LOC3/gan/ci_di_vae_gan/ConvBased/domain_and_class/gan_discriminator.keras")
