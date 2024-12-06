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


@tf.function
def train_step_ci_di(vae_model, class_discriminator, domain_discriminator, x, y, d, optimizer):
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var = vae_model(x)
        vae_loss_value = vae_loss(x, reconstructed, z_mean, z_log_var)

        # Get the latent representation z from the mean
        z = z_mean  # Optionally, you could use Sampling(z_mean, z_log_var)

        # Compute the classification loss using the class_discriminator
        class_preds = class_discriminator(z)
        class_loss_value = classification_loss(y, class_preds)

        domain_preds = domain_discriminator(z)
        domain_loss_value = classification_loss(d, domain_preds)

        tf.debugging.check_numerics(vae_loss_value, "VAE Loss")
        tf.debugging.check_numerics(class_loss_value, "Class Loss")
        tf.debugging.check_numerics(domain_loss_value, "Domain Loss")

        # Combine VAE loss with classification loss
        total_loss = vae_loss_value + class_loss_value + domain_loss_value

    grads = tape.gradient(
        total_loss, vae_model.trainable_variables + class_discriminator.trainable_variables + domain_discriminator.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, vae_model.trainable_variables + class_discriminator.trainable_variables + domain_discriminator.trainable_variables))
    return total_loss, vae_loss_value, class_loss_value, domain_loss_value


@tf.function
def train_step_ci(vae_model, class_discriminator, x, y, optimizer):
    with tf.GradientTape() as tape:
        reconstructed, z_mean, z_log_var = vae_model(x)
        vae_loss_value = vae_loss(x, reconstructed, z_mean, z_log_var)

        # Get the latent representation z from the mean
        z = z_mean  # Optionally, you could use Sampling(z_mean, z_log_var)

        # Compute the classification loss using the class_discriminator
        class_preds = class_discriminator(z)
        class_loss_value = classification_loss(y, class_preds)

        # Combine VAE loss with classification loss
        total_loss = vae_loss_value + class_loss_value

    grads = tape.gradient(
        total_loss, vae_model.trainable_variables + class_discriminator.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, vae_model.trainable_variables + class_discriminator.trainable_variables))
    return total_loss, vae_loss_value, class_loss_value


def train_ci_di_vae(vae_model, class_discriminator, domain_discriminator, train_dataset, optimizer, epochs):
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_vae_loss = tf.keras.metrics.Mean()
        epoch_class_loss = tf.keras.metrics.Mean()
        epoch_domain_loss = tf.keras.metrics.Mean()

        for step, (x, y, d) in enumerate(train_dataset):
            total_loss, vae_loss_value, class_loss_value, domain_loss_value = train_step_ci_di(
                vae_model, class_discriminator, domain_discriminator, x, y, d, optimizer)
            epoch_loss.update_state(total_loss)
            epoch_vae_loss.update_state(vae_loss_value)
            epoch_class_loss.update_state(class_loss_value)
            epoch_domain_loss.update_state(domain_loss_value)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss.result():.4f}, VAE Loss: {epoch_vae_loss.result():.4f}, Class Loss: {epoch_class_loss.result():.4f}, Domain Loss: {epoch_domain_loss.result():.4f}")

        if (epoch > 0) and (epoch % 20 == 0):
            vae_model.save(
                f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/checkpoints/LOC2-LOC3-e{epoch}-mse1-kl_mixed-cl1.0-ConvBatchNorm-ldim{latent_dim}-hdim{hidden_dim}.keras")


def train_ci_vae(vae_model, class_discriminator, train_dataset, optimizer, epochs):
    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_vae_loss = tf.keras.metrics.Mean()
        epoch_class_loss = tf.keras.metrics.Mean()

        for step, (x, y, d) in enumerate(train_dataset):
            total_loss, vae_loss_value, class_loss_value = train_step_ci(
                vae_model, class_discriminator, x, y, optimizer)
            epoch_loss.update_state(total_loss)
            epoch_vae_loss.update_state(vae_loss_value)
            epoch_class_loss.update_state(class_loss_value)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss.result():.4f}, VAE Loss: {epoch_vae_loss.result():.4f}, Class Loss: {epoch_class_loss.result():.4f}")

        if (epoch > 0) and (epoch % 20 == 0):
            if (epoch > 0) and (epoch % 20 == 0):
                vae_model.save(
                    f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/checkpoints/LOC2-LOC3-e{epoch}-mse1-kl_mixed-cl1.0-ConvBatchNorm-ldim{latent_dim}-hdim{hidden_dim}.keras")


def linear_discriminator(input_dim, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(num_classes, activation=None,
                              input_shape=(input_dim,))
    ])


def train_discriminant_for_pretrained_vae(discriminator, vae_model, x_train, y_train):
    from classification import preprocess_data_for_platform_classification

    def get_z_embeddings(data, vae_model=vae_model):
        embeddings = []
        chunk_size = 200
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            _, _, transformed_chunk = vae_model.encode(chunk)
            embeddings.append(transformed_chunk)

        return np.vstack(embeddings)

    X_train_latent = get_z_embeddings(x_train)
    discriminator.compile(optimizer='adam', loss=classification_loss)
    discriminator.fit(X_train_latent, y_train, epochs=50,
                      shuffle=True, batch_size=32)


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

    vae_model = tf.keras.models.load_model(
        f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/LOC2-LOC3-e800-mse1-kl0.01-cl1.0-ConvBatchNorm-ldim{latent_dim}-hdim{hidden_dim}.keras")
    class_discriminator = tf.keras.models.load_model(
        f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/linear_discriminators/class_discriminator-kl0.01.keras")
    linear_discriminator = tf.keras.models.load_model(
        f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/linear_discriminators/domain_discriminator-kl0.01.keras")
   # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # To continue training from a pretrained model freeze the vae model and train the discriminants first
    # train_discriminant_for_pretrained_vae(
    #     domain_discriminator, vae_model, x_train, d_train)

    # train_discriminant_for_pretrained_vae(
    #     class_discriminator, vae_model, x_train, y_train)

    # Train the model
    # vae_model.trainable = True
    # class_discriminator.trainable = True
    # domain_discriminator.trainable = True
    epochs = 100
    train_ci_di_vae(vae_model, class_discriminator, domain_discriminator,
                    train_dataset, optimizer, epochs=epochs)

    # Save the vae_model
    vae_model.save(
        f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/LOC2-LOC3-e{epochs}-mse1-kl_mixed-cl1.0-ConvBatchNorm-ldim{latent_dim}-hdim{hidden_dim}.keras")

    class_discriminator.save(
        f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/linear_discriminators/class_discriminator-kl_mixed.keras")
    domain_discriminator.save(
        f"../../models-LOC2-LOC3/vae/ci_vae/ConvBased/domain_and_class/linear_discriminators/domain_discriminator-kl_mixed.keras")
