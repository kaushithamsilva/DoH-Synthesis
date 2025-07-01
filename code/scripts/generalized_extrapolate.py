import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from init_dataset import get_train_test_dataset
from train_vae import VAE
import model_utils
from hyperplane import Hyperplane

# --- Configuration Paths ---
DATASET_CSV = '../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv'
SAVE_PATH = '../../models/binary_ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
CHECKPOINT_EPOCH = 700
DISCRIMINATOR_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae_e{CHECKPOINT_EPOCH}'

# Base output directory for extrapolated traffic samples
BASE_OUTPUT_DIR = '../../figures/traffic_extrapolation/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Model parameters (should match training configuration)
SEQUENCE_LENGTH = 32
LATENT_DIM = 8

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


def get_attribute_index(attribute_name, attr_names_list):
    """
    Helper to get the index of a specific attribute.
    """
    try:
        return attr_names_list.index(attribute_name)
    except ValueError:
        raise ValueError(
            f"Attribute '{attribute_name}' not found in attribute list. Available: {attr_names_list}")


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
    feature_cols = [str(i) for i in range(SEQUENCE_LENGTH)]

    # randomly sample from filtered_df
    selected_df = filtered_df.sample(n=num_samples, random_state=None)
    selected_sequences = selected_df[feature_cols].astype("float32").values

    return selected_sequences


def plot_traffic_sequences(sequences, titles=None, main_title="", filename="", output_dir=""):
    """
    Plots a list of network traffic sequences and saves them.
    """
    num_sequences = len(sequences)
    if num_sequences == 0:
        print("No sequences to plot.")
        return

    fig_cols = min(num_sequences, 3)
    fig_rows = int(np.ceil(num_sequences / fig_cols))

    plt.figure(figsize=(6 * fig_cols, 4 * fig_rows))
    for i, seq in enumerate(sequences):
        plt.subplot(fig_rows, fig_cols, i + 1)
        seq_data = seq.numpy() if tf.is_tensor(seq) else seq
        plt.plot(seq_data, alpha=0.8, linewidth=1.2)
        plt.xlabel('Time Step')
        plt.ylabel('Packet Count')
        plt.grid(True, alpha=0.3)
        if titles and i < len(titles):
            plt.title(titles[i])
        else:
            plt.title(f'Step {i}')

    plt.suptitle(main_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {full_path}")
    plt.show()
    plt.close()


def plot_comparison_with_actual(actual_sequence, source_sequence, synthesized_sequence, title="", filename="", output_dir=""):
    """
    Plots comparison between actual, source, and final synthesized traffic sequences.
    """
    plt.figure(figsize=(12, 8))

    # Main comparison plot
    plt.subplot(2, 1, 1)
    plt.plot(actual_sequence, label='Actual Traffic',
             alpha=0.8, linewidth=2, color='green')
    plt.plot(source_sequence, label='Source (Original)',
             alpha=0.8, linewidth=2, color='blue')
    plt.plot(synthesized_sequence, label='Synthesized (Final)',
             alpha=0.8, linewidth=2, color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Packet Count')
    plt.title(f'{title} - Traffic Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Difference plot
    plt.subplot(2, 1, 2)
    diff_actual_synth = np.array(actual_sequence) - \
        np.array(synthesized_sequence)
    diff_source_synth = np.array(source_sequence) - \
        np.array(synthesized_sequence)
    plt.plot(diff_actual_synth, label='Actual - Synthesized',
             alpha=0.8, linewidth=1.5, color='purple')
    plt.plot(diff_source_synth, label='Source - Synthesized',
             alpha=0.8, linewidth=1.5, color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Difference in Packet Count')
    plt.title('Difference Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if filename:
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {full_path}")
    plt.show()
    plt.close()


def run_location_synthesis_experiment(
    vae_model,
    discriminators,
    attribute_names,
    test_df,
    source_features,
    target_location,
    experiment_params,
    website_id
):
    """
    Synthesizes traffic from source location to target location while keeping other features fixed.
    Use website_id for both source and target.
    """
    print(
        f"\n--- Starting Location Synthesis: {source_features['location']} → {target_location} (website_id={website_id}) ---")

    experiment_name = f"{source_features['location']}_to_{target_location}_website_{website_id}"
    experiment_output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Select initial traffic sample based on source features and website_id
    print(
        f"Selecting traffic sample with source features: {source_features} and website_id: {website_id}...")
    try:
        initial_sequences = select_samples_with_specific_features(
            test_df,
            source_features,
            website_id,
            num_samples=1,
        )

        original_sequence = tf.expand_dims(initial_sequences[0], axis=0)

        plot_traffic_sequences([original_sequence[0]],
                               titles=[
                                   f"Source Traffic (Website {website_id})"],
                               main_title=f"Source: {source_features}",
                               filename="source_traffic.png",
                               output_dir=experiment_output_dir)

    except ValueError as e:
        print(f"Error selecting initial sample: {e}")
        return

    # Load discriminators for target location and fixed features
    print("Loading discriminators...")

    # Target location discriminator
    target_location_attr = f"location_{target_location}"
    target_location_discriminator = discriminators.get(target_location_attr)
    if not target_location_discriminator:
        print(
            f"Failed to load discriminator for {target_location_attr}. Exiting experiment.")
        return

    # Fixed feature discriminators
    fixed_discriminators = {}
    for feature, value in source_features.items():
        if feature != "location":  # Don't fix the location we're changing
            attr_name = f"{feature}_{value}"
            disc = discriminators.get(attr_name)
            if disc:
                fixed_discriminators[attr_name] = disc
                print(f"Loaded discriminator for fixed feature: {attr_name}")
            else:
                print(f"Warning: Could not load discriminator for {attr_name}")

    print(f"Target location discriminator: {target_location_attr}")
    print(f"Fixed feature discriminators: {list(fixed_discriminators.keys())}")

    # Extract hyperplane parameters for the target location
    target_location_hyperplane = Hyperplane(target_location_discriminator)
    direction_vector, _ = target_location_hyperplane.get_hyplerplane_params()
    direction_vector = tf.expand_dims(direction_vector, axis=0)

    print(
        f"Direction vector shape for {target_location}: {direction_vector.shape}")

    # Encode the original traffic sequence
    initial_z_mean, _, _ = vae_model.encode(original_sequence)
    current_z = tf.identity(initial_z_mean)

    # Extrapolation parameters
    num_steps = experiment_params.get('num_steps', 25)
    step_size = experiment_params.get('step_size', 0.2)
    pull_strength = experiment_params.get('pull_strength', 0.015)
    target_threshold = experiment_params.get('target_threshold', 0.5)
    fixed_threshold = experiment_params.get('fixed_threshold', 0.5)

    max_latent_norm_threshold = 10 * np.sqrt(LATENT_DIM)
    print(f"Maximum allowed latent norm: {max_latent_norm_threshold:.2f}")
    print(f"Pull-to-center strength (beta): {pull_strength}")
    print(f"Target location threshold: {target_threshold}")
    print(f"Fixed features threshold: {fixed_threshold}")

    generated_sequences = [original_sequence[0]]
    step_titles = [f"Source ({source_features['location']})"]

    print(
        f"\nStarting synthesis from {source_features['location']} to {target_location}...")

    for i in range(num_steps):
        # Get current discriminator scores
        target_score = target_location_discriminator(current_z).numpy()[0, 0]

        fixed_scores = {}
        for attr_name, disc in fixed_discriminators.items():
            fixed_scores[attr_name] = disc(current_z).numpy()[0, 0]

        current_latent_norm = tf.norm(current_z).numpy()

        # Stop condition 1: Target location threshold reached
        if target_score > target_threshold:
            print(
                f"Stopped at step {i+1}: Target location '{target_location}' threshold reached. Score: {target_score:.2f}")
            break

        # Stop condition 2: Latent vector too far from origin
        if current_latent_norm > max_latent_norm_threshold:
            print(
                f"Stopped at step {i+1}: Latent vector norm ({current_latent_norm:.2f}) exceeded threshold.")
            break

        # Stop condition 3: Any fixed feature deviates too much
        should_stop = False
        for attr_name, score in fixed_scores.items():
            if score < fixed_threshold:
                print(
                    f"Stopped at step {i+1}: Fixed feature '{attr_name}' deviated. Score: {score:.2f} < {fixed_threshold:.2f}")
                should_stop = True
                break

        if should_stop:
            break

        # Apply extrapolation step with pull-to-center
        current_z = (1 - pull_strength) * current_z + \
            step_size * direction_vector

        # Decode the new latent vector
        decoded_sequence = vae_model.decode(current_z)
        generated_sequences.append(decoded_sequence[0])
        step_titles.append(f"Step {i+1}")

        # Log progress
        log_str = f"Step {i+1}: {target_location} = {target_score:.2f}"
        for attr_name, score in fixed_scores.items():
            log_str += f", {attr_name} = {score:.2f}"
        log_str += f", Norm = {current_latent_norm:.2f}"
        print(log_str)

    print(
        f"Synthesis completed. Generated {len(generated_sequences)} traffic sequences.")

    # Visualize results
    plot_traffic_sequences(generated_sequences,
                           titles=step_titles,
                           main_title=f"Location Synthesis: {source_features['location']} → {target_location}",
                           filename=f"{experiment_name}_progression.png",
                           output_dir=experiment_output_dir)

    # Try to find actual target location traffic for comparison (same website_id)
    target_features = source_features.copy()
    target_features['location'] = target_location

    try:
        target_sequences = select_samples_with_specific_features(
            test_df, target_features, website_id, num_samples=1
        )
        plot_comparison_with_actual(target_sequences[0],
                                    generated_sequences[0].numpy(),
                                    generated_sequences[-1].numpy(),
                                    title=f"Synthesis: {source_features['location']} → {target_location} (website_id={website_id})",
                                    filename=f"{experiment_name}_comparison.png",
                                    output_dir=experiment_output_dir)

    except ValueError:
        print(
            f"Could not find actual {target_location} traffic with same features and website_id={website_id} for comparison.")
        plot_comparison_with_actual(initial_sequences[0],
                                    generated_sequences[0].numpy(),
                                    generated_sequences[-1].numpy(),
                                    title=f"Synthesis: {source_features['location']} → {target_location} (website_id={website_id})",
                                    filename=f"{experiment_name}_comparison.png",
                                    output_dir=experiment_output_dir)

    print(
        f"--- Location Synthesis Complete: {source_features['location']} → {target_location} (website_id={website_id}) ---")


def load_models_and_data():
    """Load all necessary models and data for extrapolation experiments."""
    print("Loading VAE model...")
    vae_model = model_utils.load_model(CHECKPOINT_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        raise FileNotFoundError('VAE model not found')
    print("VAE model loaded successfully.")

    print("Loading dataset...")
    train_ds, test_ds, train_website_ids, test_website_ids, attribute_names = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1, length=SEQUENCE_LENGTH)

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
    print("--- Initializing Network Traffic Location Synthesis Script ---")

    vae_model, discriminators, attribute_names, test_website_ids, test_df = load_models_and_data()

    print(f"Available attributes: {attribute_names}")
    print(f"Available discriminators: {list(discriminators.keys())}")

    experiment_params = {
        'num_steps': 100,
        'step_size': 0.1,
        'pull_strength': 0.010,
        'target_threshold': 5.0,
        'fixed_threshold': 0.2,
    }

    source_features_1 = {
        "location": "leuven",
        "client": "cloudflare",
        "resolver": "cloudflare",
        "platform": "desktop"
    }

    for _ in range(3):
        # Randomly select a website ID from the test set
        website_id = random.sample(test_website_ids, 1)[0]
        print(
            f"Running location synthesis experiment for website_id: {website_id}")

        run_location_synthesis_experiment(
            vae_model, discriminators, attribute_names,
            test_df, source_features_1, "singapore", experiment_params,
            website_id=website_id
        )

    print("\n--- All Location Synthesis Experiments Complete ---")
