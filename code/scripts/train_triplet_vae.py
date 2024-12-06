import tensorflow as tf
from triplet_functions import n_neurons
import init_gpu as init_gpu
import init_dataset as init_dataset
import random
from tensorflow import keras
import numpy as np
from triplet_vae_model import Sampling, VAE_Triplet_Model, Pretrained_Triplet_VAE
import pandas as pd


def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.maximum(basic_loss, 0.0)
    return tf.reduce_mean(loss)


@tf.function
def train_step(model, optimizer, anchor, positive, negative, platform, epoch=None):
    with tf.GradientTape() as tape:
        # Get model outputs using new architecture
        reconstructed, anchor_mean, anchor_log_var, anchor_emb, positive_emb, negative_emb = model(
            (anchor, positive, negative, platform))

        # Reconstruction loss
        mse_loss = tf.reduce_mean(
            tf.square(tf.cast(anchor, tf.float32) - reconstructed))

        # KL divergence loss (using anchor encodings)
        kl_loss = -0.5 * tf.reduce_mean(
            1 + anchor_log_var - tf.square(anchor_mean) - tf.exp(anchor_log_var))

        # Triplet loss (now using the z_means from VAE)
        triplet_loss_value = triplet_loss(
            anchor_emb, positive_emb, negative_emb)

        # Combined loss function with balanced weights
        total_loss = mse_loss + kl_loss + triplet_loss_value

    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, mse_loss, kl_loss, triplet_loss_value


def train_model(model, train_dataset, optimizer, epochs, log_every_n_steps=100):
    for epoch in range(epochs):
        for step, (anchor, positive, negative, platform) in enumerate(train_dataset):
            loss, mse_loss, kl_loss, triplet_loss_value = train_step(
                model, optimizer, anchor, positive, negative, platform)

            if step % log_every_n_steps == 0:
                print(f"Epoch {epoch+1}, Step {step}:")
                print(f"\tTotal loss: {loss:.6f}")
                print(f"\tMSE loss: {mse_loss:.6f}")
                print(f"\tKL loss: {kl_loss:.6f}")
                print(f"\tTriplet loss: {triplet_loss_value:.4f}")


def location_to_onehot(select_location, locations):
    if select_location not in locations:
        raise ValueError(
            f"'{select_location}' is not in the list of locations")

    return np.array([1 if location == select_location else 0 for location in locations]).astype(np.float32)


def get_triplets_website_encoder(df, location_to_onehot_dict, num_triplet_samples):
    # Pre-compute unique locations and websites
    unique_locations = df['Location'].unique()
    unique_websites = df['Website'].unique()

    # Create dictionaries for faster lookups
    location_to_index = {loc: i for i, loc in enumerate(unique_locations)}
    website_to_index = {web: i for i, web in enumerate(unique_websites)}

    # Convert locations and websites to integer indices
    location_indices = df['Location'].map(location_to_index).values
    website_indices = df['Website'].map(website_to_index).values

    # Pre-compute masks for each website
    website_masks = {web: website_indices ==
                     i for i, web in enumerate(unique_websites)}

    # Pre-compute data array
    data = df.iloc[:, 2:].to_numpy()

    # Initialize arrays for results
    n_samples = len(df) * num_triplet_samples
    anchors = np.zeros((n_samples, data.shape[1]))
    positives = np.zeros((n_samples, data.shape[1]))
    negatives = np.zeros((n_samples, data.shape[1]))
    anchor_location_labels = np.zeros(
        (n_samples, len(unique_locations)), dtype=np.float32)

    idx = 0
    for i in range(len(df)):
        anchor_location = location_indices[i]
        anchor_website = website_indices[i]

        # Select negative samples
        negative_mask = (location_indices != anchor_location) & (
            website_indices != anchor_website)
        negative_indices = np.where(negative_mask)[0]
        neg_samples = np.random.choice(
            negative_indices, num_triplet_samples, replace=True)

        # Select positive samples
        positive_mask = website_masks[unique_websites[anchor_website]]
        positive_indices = np.where(positive_mask)[0]
        pos_samples = np.random.choice(
            positive_indices, num_triplet_samples, replace=True)

        # Add samples to result arrays
        anchors[idx:idx+num_triplet_samples] = data[i]
        positives[idx:idx+num_triplet_samples] = data[pos_samples]
        negatives[idx:idx+num_triplet_samples] = data[neg_samples]
        anchor_location_labels[idx:idx +
                               num_triplet_samples] = location_to_onehot_dict[df['Location'][i]]
        idx += num_triplet_samples

    return anchors, positives, negatives, anchor_location_labels,


if __name__ == '__main__':

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

    location_to_onehot_dict = {location: location_to_onehot(
        location, locations) for location in locations}

    print("Generating Triplets...")
    # get triplet data
    train_anchors, train_positives, train_negatives, train_anchor_location_labels = get_triplets_website_encoder(
        train_df, location_to_onehot_dict, num_triplet_samples=5)

    print(f"Shape: {train_anchors.shape}")
    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_anchors, train_positives, train_negatives, train_anchor_location_labels
    ))

    # Shuffle and batch the dataset
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(128)

    # model parameters
    input_dim = length
    latent_dim = 32
    hidden_dim = 64
    num_locations = len(locations)
    epochs = 100
    model = Pretrained_Triplet_VAE(input_dim, latent_dim,
                                   hidden_dim, num_locations,)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_model(model, train_dataset, optimizer, epochs=epochs)

    model.save(
        f'../../models/triplet-vae/triplet_vae-e{epochs}-mse1-kl1-t1.keras')
