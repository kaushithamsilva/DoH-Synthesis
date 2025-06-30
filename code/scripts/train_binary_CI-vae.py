import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime

import model_utils  # Ensure model_utils.py has save_model
from init_dataset import get_train_test_dataset
from train_vae import ConvVAE_BatchNorm, Sampling
import init_gpu

# Initialize GPUs
init_gpu.initialize_gpus()

# --- Configuration Constants ---
SAVE_PATH = '../../models/binary_ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
DISCRIMINATOR_SAVE_PATH = os.path.join(SAVE_PATH, 'discriminators/')
EPOCH_CHECKPOINT_INTERVAL = 50
KL_WEIGHT = 0.001
CLASSIFICATION_LOSS_WEIGHT = 1.0
EPOCHS = 1000  # Adjust as needed
BATCH_SIZE = 256
LEARNING_RATE = 1e-4

# Ensure directories exist
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(DISCRIMINATOR_SAVE_PATH, exist_ok=True)

# --- Loss Function ---
attribute_classification_loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE
)

# --- Discriminator Helper ---


def linear_discriminator(input_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation=None, input_shape=(input_dim,))
    ])

# --- Training Step ---


@tf.function
def train_step_ci(vae_model, discriminators, x, y, optimizer):
    with tf.GradientTape() as tape:
        # VAE forward
        reconstructed, z_mean, z_log_var = vae_model(x, training=True)
        # Reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(x - reconstructed))
        # KL loss
        kl_loss = -0.5 * \
            tf.reduce_mean(1 + z_log_var - tf.square(z_mean) -
                           tf.exp(z_log_var))
        z = z_mean
        # Classification losses
        cls_losses = []
        for i, disc in enumerate(discriminators):
            labels = tf.expand_dims(y[:, i], axis=-1)
            logits = disc(z, training=True)
            loss_i = attribute_classification_loss_fn(labels, logits)
            cls_losses.append(tf.reduce_mean(loss_i))
        total_cls_loss = tf.add_n(cls_losses)
        # Total
        total_loss = recon_loss + KL_WEIGHT * kl_loss + \
            CLASSIFICATION_LOSS_WEIGHT * total_cls_loss
    # Gradients
    vars = vae_model.trainable_variables + \
        sum([d.trainable_variables for d in discriminators], [])
    grads = tape.gradient(total_loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return {
        'total_loss': total_loss,
        'reconstruction_loss': recon_loss,
        'kl_loss': kl_loss,
        'classification_loss': total_cls_loss
    }

# --- Training Loop ---


def train_ci_vae(vae_model, discriminators, train_ds, val_ds, optimizer, epochs, attribute_names=None):
    if attribute_names is None:
        attribute_names = [f'attr_{i}' for i in range(len(discriminators))]
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        start = datetime.datetime.now()
        # Metrics
        train_metrics = {k: tf.keras.metrics.Mean() for k in [
            'total_loss', 'reconstruction_loss', 'kl_loss', 'classification_loss']}
        val_metrics = {k: tf.keras.metrics.Mean() for k in [
            'total_loss', 'reconstruction_loss', 'kl_loss', 'classification_loss']}
        # Training
        for x_batch, y_batch in train_ds:
            losses = train_step_ci(
                vae_model, discriminators, x_batch, y_batch, optimizer)
            for k, v in losses.items():
                train_metrics[k].update_state(v)
        # Validation
        for x_batch, y_batch in val_ds:
            recon, z_mean, z_log_var = vae_model(x_batch, training=False)
            rl = tf.reduce_mean(tf.square(x_batch - recon))
            kl = -0.5 * tf.reduce_mean(1 + z_log_var -
                                       tf.square(z_mean) - tf.exp(z_log_var))
            cls_losses = []
            for i, disc in enumerate(discriminators):
                labels = tf.expand_dims(y_batch[:, i], axis=-1)
                logits = disc(z_mean, training=False)
                loss_i = attribute_classification_loss_fn(labels, logits)
                cls_losses.append(tf.reduce_mean(loss_i))
            total_cls = tf.add_n(cls_losses)
            tot = rl + KL_WEIGHT * kl + CLASSIFICATION_LOSS_WEIGHT * total_cls
            for name, m in val_metrics.items():
                if name == 'total_loss':
                    m.update_state(tot)
                if name == 'reconstruction_loss':
                    m.update_state(rl)
                if name == 'kl_loss':
                    m.update_state(kl)
                if name == 'classification_loss':
                    m.update_state(total_cls)
        end = datetime.datetime.now()
        # Logging
        print(f"Epoch {epoch+1}/{epochs} — {(end-start).total_seconds():.2f}s")
        for k, m in train_metrics.items():
            print(f"  train_{k}: {m.result().numpy():.4f}")
        for k, m in val_metrics.items():
            print(f"  val_{k}:   {m.result().numpy():.4f}")
        # Checkpoints
        if (epoch+1) % EPOCH_CHECKPOINT_INTERVAL == 0:
            print(f"Saving checkpoint at epoch {epoch+1}")
            model_utils.save_model(
                vae_model, CHECKPOINT_PATH, f'vae_e{epoch+1}')
            for i, disc in enumerate(discriminators):
                model_utils.save_model(
                    disc, CHECKPOINT_PATH, f'{attribute_names[i]}_disc_e{epoch+1}')
    # Final save
    print("Training complete — saving final models.")
    model_utils.save_model(vae_model, SAVE_PATH, 'vae_final')
    for i, disc in enumerate(discriminators):
        model_utils.save_model(disc, DISCRIMINATOR_SAVE_PATH,
                               f'{attribute_names[i]}_disc_final')


# --- Main ---
if __name__ == '__main__':
    # Load data
    DATASET_PATH = "../../dataset/processed/processed_dataset.csv"
    length = 32
    train_ds, test_ds, train_ids, test_ids, attr_names = get_train_test_dataset(
        DATASET_PATH,
        num_train=1200,
        num_test=300,
        batch_size=BATCH_SIZE,
        random_seed=42,
        length=length,
    )

    # Initialize VAE
    input_dim = length
    latent_dim = 8
    hidden_dim = 16
    vae_model = ConvVAE_BatchNorm(input_dim, latent_dim, hidden_dim)
    # Build shapes
    for x_batch, _ in train_ds.take(1):
        _ = vae_model(x_batch)
        break

    # Initialize Discriminators
    discriminators = [linear_discriminator(latent_dim) for _ in attr_names]

    # Optimizer
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE)

    # Train
    train_ci_vae(
        vae_model,
        discriminators,
        train_ds,
        test_ds,
        optimizer,
        epochs=EPOCHS,
        attribute_names=attr_names
    )
