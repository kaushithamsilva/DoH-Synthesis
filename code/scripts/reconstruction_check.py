import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from init_dataset import get_train_test_dataset
from train_vae import VAE
import model_utils

# --- Configuration ---
DATASET_CSV = '../../dataset/processed/processed_dataset.csv'
CHECKPOINT_PATH = '../../models/binary_ci_vae/checkpoints/'
CHECKPOINT_EPOCH = 150
VAE_MODEL_NAME = f'vae_e{CHECKPOINT_EPOCH}'
SEQUENCE_LENGTH = 32
NUM_SAMPLES = 6  # Number of test samples to visualize

# --- Load Data ---
print("Loading test data...")
_, test_ds, _, _, _ = get_train_test_dataset(
    DATASET_CSV, num_train=1200, num_test=300, batch_size=1, random_seed=42, length=SEQUENCE_LENGTH
)

# --- Load VAE Model ---
print("Loading VAE model...")
vae_model = model_utils.load_model(CHECKPOINT_PATH, VAE_MODEL_NAME)
if vae_model is None:
    raise FileNotFoundError("VAE model not found.")

# --- Plot Reconstructions ---


def plot_reconstructions(originals, reconstructions, num_samples=5, save_path=None):
    plt.figure(figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(originals[i], label='Original', color='blue')
        plt.plot(reconstructions[i], label='Reconstruction',
                 color='red', linestyle='--')
        plt.title(f"Sample {i+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Packet Count")
        plt.legend()
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstruction plot to {save_path}")
    plt.show()
    plt.close()


# --- Run Reconstruction Check ---
originals = []
reconstructions = []

print("Generating reconstructions...")
for idx, (x_batch, _) in enumerate(test_ds):
    if idx >= NUM_SAMPLES:
        break
    x = x_batch.numpy()[0]
    x_recon, _, _ = vae_model(x_batch, training=False)
    x_recon = x_recon.numpy()[0]
    originals.append(x)
    reconstructions.append(x_recon)

# Ensure save directory exists
save_dir = "../figures/reconstructions"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "vae_reconstructions.png")

plot_reconstructions(originals, reconstructions,
                     num_samples=NUM_SAMPLES, save_path=save_path)
