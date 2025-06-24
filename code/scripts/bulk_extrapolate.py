import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from collections import defaultdict

from init_dataset import get_train_test_dataset
from train_vae import VAE
import model_utils
from hyperplane import Hyperplane

# --- Configuration Paths ---
DATASET_CSV = '../../dataset/processed/processed_dataset.csv'
SAVE_PATH = '../../models/binary_ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
CHECKPOINT_EPOCH = 100
DISCRIMINATOR_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae_e{CHECKPOINT_EPOCH}'

# Base output directory for extrapolated traffic samples
BASE_OUTPUT_DIR = '../../figures/bulk_traffic_synthesis/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Model parameters (should match training configuration)
SEQUENCE_LENGTH = 128  # Assuming 128 time steps for network traffic
LATENT_DIM = 32  # Adjust based on your VAE configuration

# Dataset feature definitions
LOCATIONS = ["lausanne", "leuven", "singapore"]
CLIENTS = ["cloudflare", "firefox"]
RESOLVERS = ["google", "cloudflare"]
PLATFORMS = ["desktop", "desktop_(aws)", "raspberry_pi"]

# Feature to attribute mapping (adjust based on your dataset columns)
FEATURE_ATTRIBUTES = {
    "location": LOCATIONS,
    "client": CLIENTS,
    "resolver": RESOLVERS,
    "platform": PLATFORMS
}


def select_samples_with_specific_features(df, feature_criteria, website_id, num_samples=1):
    """
    Selects network traffic samples that match the specified feature criteria from a DataFrame.
    If website_id is provided, selects only from that website.
    """
    if website_id is not None:
        df = df[df["Website"] == website_id].reset_index(drop=True)

    # Build mask for feature criteria
    mask = np.ones(len(df), dtype=bool)
    for feature, value in feature_criteria.items():
        mask &= (df[feature.capitalize()] == value.capitalize())
    filtered_df = df[mask]

    if len(filtered_df) < num_samples:
        raise ValueError(
            f"Could not find {num_samples} samples with features: {feature_criteria} and website_id: {website_id}")

    # Extract features and attributes
    feature_cols = [str(i) for i in range(128)]

    # randomly sample from filtered_df
    selected_df = filtered_df.sample(n=num_samples, random_state=None)
    selected_sequences = selected_df[feature_cols].astype("float32").values

    return selected_sequences


def synthesize_single_sample(vae_model, discriminators, original_sequence, source_features, target_location, experiment_params):
    """
    Synthesizes a single traffic sample from source location to target location.
    Returns the final synthesized sequence.
    """
    # Load discriminators for target location and fixed features
    target_location_attr = f"location_{target_location}"
    target_location_discriminator = discriminators.get(target_location_attr)
    if not target_location_discriminator:
        return None

    # Fixed feature discriminators
    fixed_discriminators = {}
    for feature, value in source_features.items():
        if feature != "location":  # Don't fix the location we're changing
            attr_name = f"{feature}_{value}"
            disc = discriminators.get(attr_name)
            if disc:
                fixed_discriminators[attr_name] = disc

    # Extract hyperplane parameters for the target location
    target_location_hyperplane = Hyperplane(target_location_discriminator)
    direction_vector, _ = target_location_hyperplane.get_hyplerplane_params()
    direction_vector = tf.expand_dims(direction_vector, axis=0)

    # Encode the original traffic sequence
    initial_z_mean, _, _ = vae_model.encode(
        tf.expand_dims(original_sequence, axis=0))
    current_z = tf.identity(initial_z_mean)

    # Extrapolation parameters
    num_steps = experiment_params.get('num_steps', 25)
    step_size = experiment_params.get('step_size', 0.2)
    pull_strength = experiment_params.get('pull_strength', 0.015)
    target_threshold = experiment_params.get('target_threshold', 0.5)
    fixed_threshold = experiment_params.get('fixed_threshold', 0.5)

    max_latent_norm_threshold = 10 * np.sqrt(LATENT_DIM)

    for i in range(num_steps):
        # Get current discriminator scores
        target_score = target_location_discriminator(current_z).numpy()[0, 0]

        fixed_scores = {}
        for attr_name, disc in fixed_discriminators.items():
            fixed_scores[attr_name] = disc(current_z).numpy()[0, 0]

        current_latent_norm = tf.norm(current_z).numpy()

        # Stop conditions
        if target_score > target_threshold:
            break
        if current_latent_norm > max_latent_norm_threshold:
            break

        should_stop = False
        for attr_name, score in fixed_scores.items():
            if score < fixed_threshold:
                should_stop = True
                break
        if should_stop:
            break

        # Apply extrapolation step with pull-to-center
        current_z = (1 - pull_strength) * current_z + \
            step_size * direction_vector

    # Decode the final latent vector
    final_sequence = vae_model.decode(current_z)
    return final_sequence[0].numpy()


def collect_distribution_samples(test_df, feature_criteria, website_ids, num_samples_per_website=1):
    """
    Collect samples from multiple websites with given feature criteria.
    Returns a list of sequences.
    """
    all_sequences = []
    successful_websites = []

    for website_id in website_ids:
        try:
            sequences = select_samples_with_specific_features(
                test_df, feature_criteria, website_id, num_samples_per_website
            )
            all_sequences.extend(sequences)
            successful_websites.append(website_id)
        except ValueError:
            continue  # Skip if no samples found for this website

    return all_sequences, successful_websites


def plot_kde_distributions(source_sequences, target_sequences, synthesized_sequences,
                           source_features, target_location, output_dir, filename):
    """
    Plot KDE distributions comparing source, actual target, and synthesized sequences.
    Creates subplots for different statistical measures.
    """
    # Calculate statistical measures for each sequence type
    def calculate_stats(sequences):
        stats = {
            'mean': [np.mean(seq) for seq in sequences],
            'std': [np.std(seq) for seq in sequences],
            'max': [np.max(seq) for seq in sequences],
            'sum': [np.sum(seq) for seq in sequences]
        }
        return stats

    source_stats = calculate_stats(source_sequences)
    target_stats = calculate_stats(target_sequences)
    synth_stats = calculate_stats(synthesized_sequences)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    stat_names = ['mean', 'std', 'max', 'sum']
    colors = ['blue', 'green', 'red']
    labels = [f"Source ({source_features['location']})",
              f"Actual Target ({target_location})", f"Synthesized ({target_location})"]

    for i, stat in enumerate(stat_names):
        ax = axes[i]

        # Plot KDE for each distribution
        data_sets = [source_stats[stat], target_stats[stat], synth_stats[stat]]

        for j, (data, color, label) in enumerate(zip(data_sets, colors, labels)):
            if len(data) > 1:  # Need at least 2 points for KDE
                try:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(min(data) - np.std(data),
                                          max(data) + np.std(data), 100)
                    ax.plot(x_range, kde(x_range), color=color,
                            label=label, alpha=0.7, linewidth=2)
                    ax.fill_between(x_range, kde(x_range),
                                    alpha=0.3, color=color)
                except:
                    # Fallback to histogram if KDE fails
                    ax.hist(data, bins=10, alpha=0.3, color=color,
                            label=label, density=True)
            else:
                # Single point - just mark it
                ax.axvline(x=data[0], color=color, label=label, linewidth=2)

        ax.set_title(f'Distribution of Sequence {stat.capitalize()}')
        ax.set_xlabel(f'Sequence {stat.capitalize()}')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Traffic Distribution Analysis: {source_features["location"]} → {target_location}',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"Saved KDE distribution plot to {full_path}")
    plt.close()


def run_bulk_synthesis_experiment(vae_model, discriminators, attribute_names, test_df,
                                  test_website_ids, source_features, target_location,
                                  experiment_params, num_websites=None):
    """
    Run bulk synthesis experiment for multiple websites.
    """
    print(
        f"\n--- Starting Bulk Synthesis: {source_features['location']} → {target_location} ---")

    # Select websites to process
    if num_websites is None:
        selected_websites = test_website_ids
    else:
        selected_websites = random.sample(
            test_website_ids, min(num_websites, len(test_website_ids)))

    print(f"Processing {len(selected_websites)} websites...")

    experiment_name = f"bulk_{source_features['location']}_to_{target_location}"
    experiment_output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Collect source sequences
    print("Collecting source sequences...")
    source_sequences, successful_source_websites = collect_distribution_samples(
        test_df, source_features, selected_websites, num_samples_per_website=1
    )

    # Collect actual target sequences
    print("Collecting actual target sequences...")
    target_features = source_features.copy()
    target_features['location'] = target_location
    target_sequences, successful_target_websites = collect_distribution_samples(
        test_df, target_features, selected_websites, num_samples_per_website=1
    )

    # Generate synthesized sequences
    print("Generating synthesized sequences...")
    synthesized_sequences = []
    successful_synth_websites = []

    for website_id in successful_source_websites:
        try:
            # Get source sequence for this website
            source_seq = select_samples_with_specific_features(
                test_df, source_features, website_id, num_samples=1
            )[0]

            # Synthesize
            synth_seq = synthesize_single_sample(
                vae_model, discriminators, source_seq, source_features,
                target_location, experiment_params
            )

            if synth_seq is not None:
                synthesized_sequences.append(synth_seq)
                successful_synth_websites.append(website_id)

        except Exception as e:
            print(f"Failed to synthesize for website {website_id}: {e}")
            continue

    # Report statistics
    print(f"Successfully collected {len(source_sequences)} source sequences")
    print(f"Successfully collected {len(target_sequences)} target sequences")
    print(f"Successfully synthesized {len(synthesized_sequences)} sequences")

    # Create KDE distribution plots
    if len(source_sequences) > 0 and len(target_sequences) > 0 and len(synthesized_sequences) > 0:
        plot_kde_distributions(
            source_sequences, target_sequences, synthesized_sequences,
            source_features, target_location, experiment_output_dir,
            f"{experiment_name}_kde_distributions.png"
        )
    else:
        print("Insufficient data for KDE plotting")

    # Save summary statistics
    summary_stats = {
        'source_websites': successful_source_websites,
        'target_websites': successful_target_websites,
        'synthesized_websites': successful_synth_websites,
        'num_source_sequences': len(source_sequences),
        'num_target_sequences': len(target_sequences),
        'num_synthesized_sequences': len(synthesized_sequences)
    }

    summary_df = pd.DataFrame([summary_stats])
    summary_path = os.path.join(
        experiment_output_dir, f"{experiment_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to {summary_path}")

    print(
        f"--- Bulk Synthesis Complete: {source_features['location']} → {target_location} ---")

    return {
        'source_sequences': source_sequences,
        'target_sequences': target_sequences,
        'synthesized_sequences': synthesized_sequences,
        'summary_stats': summary_stats
    }


def load_models_and_data():
    """Load all necessary models and data for extrapolation experiments."""
    print("Loading VAE model...")
    vae_model = model_utils.load_model(CHECKPOINT_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        raise FileNotFoundError('VAE model not found')
    print("VAE model loaded successfully.")

    print("Loading dataset...")
    train_ds, test_ds, train_website_ids, test_website_ids, attribute_names = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1)

    # Load original dataset for actual traffic sequences
    df_original = pd.read_csv(DATASET_CSV, index_col=0)

    # filter the df to only include test websites
    test_df = df_original[df_original["Website"].isin(
        test_website_ids)].reset_index(drop=True)

    print("Loading discriminator models...")
    discriminators = {}
    for attr in attribute_names:
        disc_name = f'{attr}_disc_e{CHECKPOINT_EPOCH}'
        disc = model_utils.load_model(DISCRIMINATOR_PATH, disc_name)
        if disc:
            discriminators[attr] = disc
            print(f"Loaded discriminator for {attr}")
        else:
            print(f"Warning: Could not load discriminator for {attr}")

    return vae_model, discriminators, attribute_names, test_website_ids, test_df


if __name__ == "__main__":
    print("--- Initializing Bulk Network Traffic Synthesis Script ---")

    vae_model, discriminators, attribute_names, test_website_ids, test_df = load_models_and_data()

    print(f"Available attributes: {attribute_names}")
    print(f"Available discriminators: {list(discriminators.keys())}")
    print(f"Total test websites: {len(test_website_ids)}")

    experiment_params = {
        'num_steps': 100,
        'step_size': 0.1,
        'pull_strength': 0.015,
        'target_threshold': 5.0,
        'fixed_threshold': 0.2,
    }

    source_features_1 = {
        "location": "lausanne",
        "client": "cloudflare",
        "resolver": "cloudflare",
        "platform": "desktop"
    }

    # Run bulk synthesis experiments
    results_all = run_bulk_synthesis_experiment(
        vae_model, discriminators, attribute_names, test_df, test_website_ids,
        source_features_1, "leuven", experiment_params, num_websites=50
    )

    print("\n--- All Bulk Synthesis Experiments Complete ---")
