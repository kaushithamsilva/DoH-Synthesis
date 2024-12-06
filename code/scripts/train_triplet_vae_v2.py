import tensorflow as tf
from triplet_functions import n_neurons
import init_gpu as init_gpu
import init_dataset as init_dataset
import random
from tensorflow import keras
import numpy as np
from triplet_vae_model import Sampling, VAE_Triplet_Model_V2
import pandas as pd


def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.maximum(basic_loss, 0.0)
    return tf.reduce_mean(loss)


@tf.function
def train_step(model, optimizer, anchor, positive, negative, anchor_location, positive_location, epoch=None):
    with tf.GradientTape() as tape:
        # Get model outputs using new architecture
        outputs = model((anchor, positive, negative,
                        anchor_location, positive_location))

        # Unpack reconstructions and embeddings
        recon_anchor_anchor = outputs['recon_anchor_anchor']
        recon_anchor_positive = outputs['recon_anchor_positive']
        recon_positive_anchor = outputs['recon_positive_anchor']
        recon_positive_positive = outputs['recon_positive_positive']

        # Calculate reconstruction losses
        mse_loss_aa = tf.reduce_mean(
            tf.square(tf.cast(anchor, tf.float32) - recon_anchor_anchor))
        mse_loss_ap = tf.reduce_mean(
            tf.square(tf.cast(positive, tf.float32) - recon_anchor_positive))
        mse_loss_pa = tf.reduce_mean(
            tf.square(tf.cast(anchor, tf.float32) - recon_positive_anchor))
        mse_loss_pp = tf.reduce_mean(
            tf.square(tf.cast(positive, tf.float32) - recon_positive_positive))

        # Combined reconstruction loss, higher weight for cross domain synthesis
        mse_loss = 0.1 * mse_loss_aa + 0.4 * mse_loss_ap + \
            0.4 * mse_loss_pa + 0.1 * mse_loss_pp

        # KL divergence losses for anchor, positive, and negative
        kl_loss_anchor = -0.5 * tf.reduce_mean(
            1 + outputs['anchor_log_var'] - tf.square(outputs['anchor_mean']) -
            tf.exp(outputs['anchor_log_var']))
        kl_loss_positive = -0.5 * tf.reduce_mean(
            1 + outputs['positive_log_var'] - tf.square(outputs['positive_mean']) -
            tf.exp(outputs['positive_log_var']))
        kl_loss_negative = -0.5 * tf.reduce_mean(
            1 + outputs['negative_log_var'] - tf.square(outputs['negative_mean']) -
            tf.exp(outputs['negative_log_var']))

        # Combined KL loss (average of all three)
        kl_loss = (kl_loss_anchor + kl_loss_positive + kl_loss_negative) / 3.0

        # Triplet loss using the means
        triplet_loss_value = triplet_loss(
            outputs['anchor_mean'], outputs['positive_mean'], outputs['negative_mean'])

        # Combined loss function with balanced weights
        total_loss = mse_loss * 10 + kl_loss + 100 * triplet_loss_value

    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'kl_loss': kl_loss,
        'kl_loss_anchor': kl_loss_anchor,
        'kl_loss_positive': kl_loss_positive,
        'kl_loss_negative': kl_loss_negative,
        'triplet_loss': triplet_loss_value,
        'reconstruction_losses': {
            'anchor_anchor': mse_loss_aa,
            'anchor_positive': mse_loss_ap,
            'positive_anchor': mse_loss_pa,
            'positive_positive': mse_loss_pp
        }
    }


def train_model(model, train_dataset, optimizer, epochs, log_every_n_steps=100):
    for epoch in range(epochs):
        for step, (anchor, positive, negative, anchor_location, positive_location) in enumerate(train_dataset):
            losses = train_step(
                model, optimizer, anchor, positive, negative,
                anchor_location, positive_location)

            if step % log_every_n_steps == 0:
                print(f"Epoch {epoch+1}, Step {step}:")
                print(f"\tTotal loss: {losses['total_loss']:.6f}")
                print(f"\tMSE loss: {losses['mse_loss']:.6f}")
                print(f"\tKL loss: {losses['kl_loss']:.6f}")
                print(f"\t  - Anchor KL: {losses['kl_loss_anchor']:.6f}")
                print(f"\t  - Positive KL: {losses['kl_loss_positive']:.6f}")
                print(f"\t  - Negative KL: {losses['kl_loss_negative']:.6f}")
                print(f"\tTriplet loss: {losses['triplet_loss']:.4f}")
                print("\tReconstruction losses:")
                for name, value in losses['reconstruction_losses'].items():
                    print(f"\t  - {name}: {value:.6f}")


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
    positive_location_labels = np.zeros(
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

        # Select positive samples (same website, different location)
        positive_mask = website_masks[unique_websites[anchor_website]] & (
            location_indices != anchor_location)
        positive_indices = np.where(positive_mask)[0]

        # If no positive samples with different location exist, use same location
        if len(positive_indices) == 0:
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
        positive_location_labels[idx:idx +
                                 num_triplet_samples] = [location_to_onehot_dict[location] for location in df['Location'][pos_samples]]

        idx += num_triplet_samples

    return anchors, positives, negatives, anchor_location_labels, positive_location_labels


def location_to_onehot(select_location, locations):
    if select_location not in locations:
        raise ValueError(
            f"'{select_location}' is not in the list of locations")

    return np.array([1 if location == select_location else 0 for location in locations]).astype(np.float32)


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
    train_anchors, train_positives, train_negatives, train_anchor_location_labels, train_positive_location_labels = get_triplets_website_encoder(
        train_df, location_to_onehot_dict, num_triplet_samples=5)

    print(f"Shape: {train_anchors.shape}")
    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_anchors, train_positives, train_negatives, train_anchor_location_labels, train_positive_location_labels
    ))

    # Shuffle and batch the dataset
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(128)

    # model parameters
    input_dim = length
    latent_dim = 32
    hidden_dim = 64
    num_locations = len(locations)
    epochs = 500
    model = VAE_Triplet_Model_V2(input_dim, latent_dim,
                                 hidden_dim, num_locations,)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    train_model(model, train_dataset, optimizer, epochs=epochs)

    model.save(
        f'../../models/triplet-vae/triplet_vae-v2-e{epochs}-mse10-kl1-t100-CD4.keras')
