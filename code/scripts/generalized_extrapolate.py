import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from init_dataset import get_train_test_dataset
from train_vae import VAE
import model_utils
from hyperplane import Hyperplane

# --- Configuration ---
DATASET_CSV = '../../dataset/processed/processed_dataset.csv'
SAVE_PATH = '../../models/binary_ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
CHECKPOINT_EPOCH = 100
DISCRIMINATOR_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae_e{CHECKPOINT_EPOCH}'

BASE_OUTPUT_DIR = '../../figures/traffic_extrapolation/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Extrapolation hyperparams defaults
DEFAULT_STEPS = 20
DEFAULT_STEP_SIZE = 0.3
DEFAULT_PULL_BETA = 0.02

# --- Helpers ---


def load_models():
    # Load VAE
    vae = model_utils.load_model(SAVE_PATH, VAE_MODEL_NAME)
    if vae is None:
        raise FileNotFoundError('VAE model not found')

    # Load discriminators
    # attribute_names from dataset loader
    _, _, _, _, attribute_names = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1)
    discs = {}
    for attr in attribute_names:
        disc = model_utils.load_model(
            DISCRIMINATOR_PATH, f'{attr}_e{CHECKPOINT_EPOCH}')
        if disc:
            discs[attr] = disc
    return vae, discs, attribute_names


def select_initial_sample(test_ds, attribute_names, criteria, max_batches=1000):
    # criteria: dict of attribute_name->0/1
    for x_batch, y_batch in test_ds.take(max_batches):
        mask = tf.ones((x_batch.shape[0],), tf.bool)
        for attr, val in criteria.items():
            idx = attribute_names.index(attr)
            mask &= (tf.cast(y_batch[:, idx], tf.int32) == val)
        if tf.reduce_any(mask):
            i = tf.argmax(tf.cast(mask, tf.int32)).numpy()
            return x_batch[i:i+1]
    return None


def plot_traffic(sequences, title, filename):
    plt.figure(figsize=(8, 4))
    for seq in sequences:
        plt.plot(seq, alpha=0.5)
    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Packet count')
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, filename)
    plt.savefig(out)
    plt.close()
    print(f'Saved plot: {out}')


# --- Main Extrapolation Function ---

def run_extrapolation(
    criteria: dict,
    target_attr: str,
    steps: int = DEFAULT_STEPS,
    step_size: float = DEFAULT_STEP_SIZE,
    pull_beta: float = DEFAULT_PULL_BETA
):
    # Load models and data
    vae, discs, attribute_names = load_models()
    _, test_ds, _, _, _ = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1)

    # Select initial sample
    x0 = select_initial_sample(test_ds, attribute_names, criteria)
    if x0 is None:
        print('No sample matches criteria:', criteria)
        return

    # Encode
    z0, _, _ = vae.encode(x0)
    z = tf.identity(z0)

    # Get hyperplane for target attribute
    if target_attr not in discs:
        print(f'Discriminator for {target_attr} not found')
        return
    hyper = Hyperplane(discs[target_attr])
    normal, _ = hyper.get_hyplerplane_params()
    direction = tf.expand_dims(normal, 0)

    # Extrapolate
    sequences = []
    for i in range(steps):
        # Decode
        seq = vae.decode(z)[0].numpy()
        sequences.append(seq)
        # Update z
        z = (1 - pull_beta) * z + step_size * direction

    # Plot
    title = f'Traffic Extrapolation: {target_attr}'
    filename = f'traffic_{target_attr}.png'
    plot_traffic(sequences, title, filename)


# --- CLI Support ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Network Traffic Latent Extrapolation')
    parser.add_argument('--criteria', nargs='+', metavar='ATTR=VAL',
                        help='Attribute criteria, e.g. location_lausanne=1 resolver_cloudflare=1')
    parser.add_argument('--target', required=True,
                        help='Target attribute to extrapolate')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--step-size', type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument('--beta', type=float, default=DEFAULT_PULL_BETA)
    args = parser.parse_args()

    # Parse criteria
    crit = {}
    for pair in args.criteria or []:
        key, val = pair.split('=')
        crit[key] = int(val)

    run_extrapolation(crit, args.target, args.steps, args.step_size, args.beta)
