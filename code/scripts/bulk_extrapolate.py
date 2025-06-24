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


def get_all_samples_for_website(df, feature_criteria, website_id):
    """
    Gets ALL samples for a specific website that match the feature criteria.
    Returns all matching sequences, not just a random sample.
    """
    # Filter by website first
    website_df = df[df["Website"] == website_id].reset_index(drop=True)

    if len(website_df) == 0:
        raise ValueError(f"No samples found for website_id: {website_id}")

    # Build mask for feature criteria
    mask = np.ones(len(website_df), dtype=bool)
    for feature, value in feature_criteria.items():
        mask &= (website_df[feature.capitalize()] == value.capitalize())

    filtered_df = website_df[mask]

    if len(filtered_df) == 0:
        raise ValueError(
            f"No samples found for website_id: {website_id} with features: {feature_criteria}")

    # Extract all matching sequences
    feature_cols = [str(i) for i in range(128)]
    all_sequences = filtered_df[feature_cols].astype("float32").values

    return all_sequences


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


def plot_kde_distributions_for_website(source_sequences, target_sequences, synthesized_sequences,
                                       source_features, target_location, website_id, output_dir):
    """
    Plot KDE distributions comparing source, actual target, and synthesized sequences for a single website.
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
    target_stats = calculate_stats(target_sequences) if len(
        target_sequences) > 0 else None
    synth_stats = calculate_stats(synthesized_sequences)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    stat_names = ['mean', 'std', 'max', 'sum']

    for i, stat in enumerate(stat_names):
        ax = axes[i]

        # Plot source distribution
        source_data = source_stats[stat]
        if len(source_data) > 1:
            try:
                kde = gaussian_kde(source_data)
                x_min, x_max = min(source_data), max(source_data)
                x_range = np.linspace(x_min - 0.1 * (x_max - x_min),
                                      x_max + 0.1 * (x_max - x_min), 100)
                ax.plot(x_range, kde(x_range), color='blue',
                        label=f"Source ({source_features['location']})", alpha=0.7, linewidth=2)
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color='blue')
            except:
                ax.hist(source_data, bins=min(15, len(source_data)//2 + 1), alpha=0.3,
                        color='blue', label=f"Source ({source_features['location']})", density=True)

        # Plot synthesized distribution
        synth_data = synth_stats[stat]
        if len(synth_data) > 1:
            try:
                kde = gaussian_kde(synth_data)
                x_min, x_max = min(synth_data), max(synth_data)
                x_range = np.linspace(x_min - 0.1 * (x_max - x_min),
                                      x_max + 0.1 * (x_max - x_min), 100)
                ax.plot(x_range, kde(x_range), color='red',
                        label=f"Synthesized ({target_location})", alpha=0.7, linewidth=2)
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color='red')
            except:
                ax.hist(synth_data, bins=min(15, len(synth_data)//2 + 1), alpha=0.3,
                        color='red', label=f"Synthesized ({target_location})", density=True)

        # Plot target distribution (if available)
        if target_stats and len(target_stats[stat]) > 0:
            target_data = target_stats[stat]
            if len(target_data) > 1:
                try:
                    kde = gaussian_kde(target_data)
                    x_min, x_max = min(target_data), max(target_data)
                    x_range = np.linspace(x_min - 0.1 * (x_max - x_min),
                                          x_max + 0.1 * (x_max - x_min), 100)
                    ax.plot(x_range, kde(x_range), color='green',
                            label=f"Actual Target ({target_location})", alpha=0.7, linewidth=2)
                    ax.fill_between(x_range, kde(x_range),
                                    alpha=0.3, color='green')
                except:
                    ax.hist(target_data, bins=min(15, len(target_data)//2 + 1), alpha=0.3,
                            color='green', label=f"Actual Target ({target_location})", density=True)

        ax.set_title(f'Distribution of Sequence {stat.capitalize()}')
        ax.set_xlabel(f'Sequence {stat.capitalize()}')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Website {website_id}: Traffic Distribution Analysis\n{source_features["location"]} → {target_location}',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"website_{website_id}_kde_distributions.png"
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    print(f"Saved KDE distribution plot to {full_path}")
    plt.close()


def process_single_website(vae_model, discriminators, test_df, website_id,
                           source_features, target_location, experiment_params, output_dir):
    """
    Process a single website: get all source samples, synthesize all of them, 
    get all target samples, and create KDE comparison plots.
    """
    print(f"\n--- Processing Website {website_id} ---")

    try:
        # Get ALL source samples for this website
        print(f"Collecting all source samples for website {website_id}...")
        source_sequences = get_all_samples_for_website(
            test_df, source_features, website_id)
        print(f"Found {len(source_sequences)} source samples")

        # Get ALL target samples for this website (if available)
        target_features = source_features.copy()
        target_features['location'] = target_location
        try:
            target_sequences = get_all_samples_for_website(
                test_df, target_features, website_id)
            print(f"Found {len(target_sequences)} target samples")
        except ValueError:
            print(
                f"No target samples found for website {website_id} with target features")
            target_sequences = []

        # Synthesize ALL source samples
        print(f"Synthesizing {len(source_sequences)} samples...")
        synthesized_sequences = []

        for i, source_seq in enumerate(source_sequences):
            try:
                synth_seq = synthesize_single_sample(
                    vae_model, discriminators, source_seq, source_features,
                    target_location, experiment_params
                )

                if synth_seq is not None:
                    synthesized_sequences.append(synth_seq)
                    if (i + 1) % 10 == 0:  # Progress update every 10 samples
                        print(
                            f"  Synthesized {i + 1}/{len(source_sequences)} samples")
                else:
                    print(f"  Failed to synthesize sample {i + 1}")

            except Exception as e:
                print(f"  Error synthesizing sample {i + 1}: {e}")
                continue

        print(
            f"Successfully synthesized {len(synthesized_sequences)}/{len(source_sequences)} samples")

        # Create KDE plots comparing the three distributions
        if len(synthesized_sequences) > 0:
            plot_kde_distributions_for_website(
                source_sequences, target_sequences, synthesized_sequences,
                source_features, target_location, website_id, output_dir
            )

            # Save summary statistics for this website
            summary_stats = {
                'website_id': website_id,
                'num_source_samples': len(source_sequences),
                'num_target_samples': len(target_sequences),
                'num_synthesized_samples': len(synthesized_sequences),
                'synthesis_success_rate': len(synthesized_sequences) / len(source_sequences) * 100
            }

            return summary_stats
        else:
            print(f"No synthesized samples generated for website {website_id}")
            return None

    except Exception as e:
        print(f"Error processing website {website_id}: {e}")
        return None


def run_bulk_synthesis_experiment(vae_model, discriminators, attribute_names, test_df,
                                  test_website_ids, source_features, target_location,
                                  experiment_params, num_websites=None):
    """
    Run bulk synthesis experiment for multiple websites.
    For each website, process ALL samples and create KDE comparisons.
    """
    print(
        f"\n--- Starting Bulk Synthesis: {source_features['location']} → {target_location} ---")

    # Select websites to process
    if num_websites is None:
        selected_websites = test_website_ids
        print(f"Processing ALL {len(selected_websites)} websites...")
    else:
        selected_websites = random.sample(
            test_website_ids, min(num_websites, len(test_website_ids)))
        print(
            f"Processing randomly selected {len(selected_websites)} websites...")

    experiment_name = f"bulk_{source_features['location']}_to_{target_location}"
    experiment_output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Process each website
    all_website_stats = []
    successful_websites = 0

    for i, website_id in enumerate(selected_websites):
        print(
            f"\n=== Processing website {i+1}/{len(selected_websites)}: {website_id} ===")

        website_stats = process_single_website(
            vae_model, discriminators, test_df, website_id,
            source_features, target_location, experiment_params, experiment_output_dir
        )

        if website_stats:
            all_website_stats.append(website_stats)
            successful_websites += 1

        print(
            f"Completed website {website_id} ({i+1}/{len(selected_websites)})")

    # Save overall summary
    if all_website_stats:
        summary_df = pd.DataFrame(all_website_stats)
        summary_path = os.path.join(
            experiment_output_dir, f"{experiment_name}_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # Print overall statistics
        print(f"\n--- Experiment Summary ---")
        print(f"Total websites processed: {len(selected_websites)}")
        print(f"Successful websites: {successful_websites}")
        print(
            f"Average samples per website: {summary_df['num_source_samples'].mean():.1f}")
        print(
            f"Average synthesis success rate: {summary_df['synthesis_success_rate'].mean():.1f}%")
        print(
            f"Total synthesized samples: {summary_df['num_synthesized_samples'].sum()}")

        print(f"Summary saved to: {summary_path}")

    print(
        f"--- Bulk Synthesis Complete: {source_features['location']} → {target_location} ---")

    return all_website_stats


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
        'step_size': 0.2,
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
        source_features_1, "leuven", experiment_params, num_websites=10
    )

    print("\n--- All Bulk Synthesis Experiments Complete ---")
