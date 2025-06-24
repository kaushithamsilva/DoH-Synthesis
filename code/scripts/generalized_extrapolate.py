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
DATASET_CSV = '../../dataset/processed/processed_dataset.csv'
SAVE_PATH = '../../models/binary_ci_vae/'
CHECKPOINT_PATH = os.path.join(SAVE_PATH, 'checkpoints/')
CHECKPOINT_EPOCH = 100
DISCRIMINATOR_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae_e{CHECKPOINT_EPOCH}'

# Base output directory for extrapolated traffic samples
BASE_OUTPUT_DIR = '../../figures/traffic_extrapolation/'
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


def get_attribute_index(attribute_name, attr_names_list):
    """
    Helper to get the index of a specific attribute.
    """
    try:
        return attr_names_list.index(attribute_name)
    except ValueError:
        raise ValueError(
            f"Attribute '{attribute_name}' not found in attribute list. Available: {attr_names_list}")


def select_samples_with_specific_features(dataset, website_ids, attr_names_list, feature_criteria, num_samples=1):
    """
    Selects network traffic samples that match the specified feature criteria.
    Args:
        dataset (tf.data.Dataset): The dataset to search.
        website_ids (list): List of website IDs corresponding to dataset samples.
        attr_names_list (list): List of all attribute names for index lookup.
        feature_criteria (dict): Dictionary with feature values. 
                                E.g., {"location": "lausanne", "client": "cloudflare", 
                                      "resolver": "cloudflare", "platform": "desktop"}
        num_samples (int): Number of samples to find.
    Returns:
        tuple: (selected traffic sequences, selected attribute tensors, website IDs)
    """
    selected_sequences = []
    selected_attributes_full = []
    selected_website_ids = []

    # Convert feature criteria to binary attribute criteria
    attribute_criteria = {}
    for feature, value in feature_criteria.items():
        # For each feature, set the specific value to 1, others to 0
        if feature in FEATURE_ATTRIBUTES:
            for possible_value in FEATURE_ATTRIBUTES[feature]:
                attr_name = f"{feature}_{possible_value}"
                if attr_name in attr_names_list:
                    attribute_criteria[attr_name] = 1 if possible_value == value else 0

    print(
        f"Converted feature criteria to binary attributes: {attribute_criteria}")

    # Prepare attribute indices and desired values from criteria
    attribute_indices_to_check = []
    desired_values_for_check = []
    for attr_name, attr_value in attribute_criteria.items():
        try:
            attr_idx = get_attribute_index(attr_name, attr_names_list)
            attribute_indices_to_check.append(attr_idx)
            desired_values_for_check.append(int(attr_value))
        except ValueError:
            print(
                f"Warning: Attribute '{attr_name}' not found in dataset, skipping...")
            continue

    if not attribute_indices_to_check:
        raise ValueError(
            "No valid attribute criteria found for sample selection.")

    batch_idx = 0
    for x_batch, y_batch in dataset:
        batch_mask = tf.constant(True, shape=(tf.shape(x_batch)[0],))

        for idx, desired_val in zip(attribute_indices_to_check, desired_values_for_check):
            attr_mask = (tf.cast(y_batch[:, idx], tf.int32) == desired_val)
            batch_mask = tf.logical_and(batch_mask, attr_mask)

        sequences_matching = tf.boolean_mask(x_batch, batch_mask)
        attrs_matching = tf.boolean_mask(y_batch, batch_mask)

        # Get corresponding website IDs
        batch_start_idx = batch_idx * x_batch.shape[0]
        matching_indices = tf.where(batch_mask).numpy().flatten()

        for i in range(tf.shape(sequences_matching)[0]):
            selected_sequences.append(sequences_matching[i])
            selected_attributes_full.append(attrs_matching[i])
            # Get the corresponding website ID
            actual_idx = batch_start_idx + matching_indices[i]
            if actual_idx < len(website_ids):
                selected_website_ids.append(website_ids[actual_idx])

            if len(selected_sequences) >= num_samples:
                print(
                    f"Found {len(selected_sequences)} traffic samples matching criteria.")
                return selected_sequences[:num_samples], selected_attributes_full[:num_samples], selected_website_ids[:num_samples]

        batch_idx += 1

    if len(selected_sequences) < num_samples:
        criteria_str = ", ".join(
            [f"{feature}={value}" for feature, value in feature_criteria.items()])
        raise ValueError(
            f"Could not find {num_samples} traffic samples with features: {criteria_str}.")

    return selected_sequences[:num_samples], selected_attributes_full[:num_samples], selected_website_ids[:num_samples]


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
    test_ds,
    test_website_ids,
    df_original,
    source_features,
    target_location,
    experiment_params
):
    """
    Synthesizes traffic from source location to target location while keeping other features fixed.

    Args:
        vae_model (tf.keras.Model): The loaded VAE model.
        discriminators (dict): Dictionary of loaded discriminator models.
        attribute_names (list): List of all attribute names.
        test_ds (tf.data.Dataset): The test dataset for sample selection.
        test_website_ids (list): List of website IDs corresponding to test samples.
        df_original (pd.DataFrame): Original dataset for getting actual traffic sequences.
        source_features (dict): Source feature configuration.
        target_location (str): Target location to synthesize.
        experiment_params (dict): Experiment configuration parameters.
    """
    print(
        f"\n--- Starting Location Synthesis: {source_features['location']} → {target_location} ---")

    # Set up experiment-specific output directory
    experiment_name = f"{source_features['location']}_to_{target_location}"
    experiment_output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Select initial traffic sample based on source features
    print(
        f"Selecting traffic sample with source features: {source_features}...")
    try:
        initial_sequences, _, website_ids = select_samples_with_specific_features(
            test_ds,
            test_website_ids,
            attribute_names,
            source_features,
            num_samples=1
        )

        original_sequence = tf.expand_dims(initial_sequences[0], axis=0)
        website_id = website_ids[0]

        # Get actual traffic sequence from original dataset
        actual_row = df_original[df_original['Website'] == website_id]
        packet_cols = [str(i) for i in range(SEQUENCE_LENGTH)]
        actual_sequence = actual_row.iloc[0][packet_cols].values.astype(float)

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

    # Try to find actual target location traffic for comparison
    target_features = source_features.copy()
    target_features['location'] = target_location

    try:
        target_sequences, _, target_website_ids = select_samples_with_specific_features(
            test_ds, test_website_ids, attribute_names, target_features, num_samples=1
        )
        target_actual_row = df_original[df_original['Website']
                                        == target_website_ids[0]]
        target_actual_sequence = target_actual_row.iloc[0][packet_cols].values.astype(
            float)

        plot_comparison_with_actual(target_actual_sequence,
                                    generated_sequences[0].numpy(),
                                    generated_sequences[-1].numpy(),
                                    title=f"Synthesis: {source_features['location']} → {target_location}",
                                    filename=f"{experiment_name}_comparison.png",
                                    output_dir=experiment_output_dir)

    except ValueError:
        print(
            f"Could not find actual {target_location} traffic with same features for comparison.")
        # Still plot source vs synthesized
        plot_comparison_with_actual(actual_sequence,
                                    generated_sequences[0].numpy(),
                                    generated_sequences[-1].numpy(),
                                    title=f"Synthesis: {source_features['location']} → {target_location}",
                                    filename=f"{experiment_name}_comparison.png",
                                    output_dir=experiment_output_dir)

    print(
        f"--- Location Synthesis Complete: {source_features['location']} → {target_location} ---")


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

    return vae_model, discriminators, attribute_names, test_ds, test_website_ids, df_original


if __name__ == "__main__":
    print("--- Initializing Network Traffic Location Synthesis Script ---")

    try:
        # Load all models and data
        vae_model, discriminators, attribute_names, test_ds, test_website_ids, df_original = load_models_and_data()

        print(f"Available attributes: {attribute_names}")
        print(f"Available discriminators: {list(discriminators.keys())}")

        # Experiment parameters
        experiment_params = {
            'num_steps': 30,
            'step_size': 0.2,
            'pull_strength': 0.015,
            'target_threshold': 0.6,  # Stop when target location score > 0.6
            'fixed_threshold': 0.4,   # Stop if any fixed feature score < 0.4
        }

        # Example 1: Lausanne → Leuven
        source_features_1 = {
            "location": "lausanne",
            "client": "cloudflare",
            "resolver": "cloudflare",
            "platform": "desktop"
        }

        run_location_synthesis_experiment(
            vae_model, discriminators, attribute_names, test_ds,
            test_website_ids, df_original, source_features_1, "leuven", experiment_params)

        # Example 2: Lausanne → Singapore
        source_features_2 = {
            "location": "lausanne",
            "client": "firefox",
            "resolver": "google",
            "platform": "desktop"
        }

        run_location_synthesis_experiment(
            vae_model, discriminators, attribute_names, test_ds,
            test_website_ids, df_original, source_features_2, "singapore", experiment_params)

        # Example 3: Singapore → Leuven
        source_features_3 = {
            "location": "singapore",
            "client": "cloudflare",
            "resolver": "google",
            "platform": "raspberry_pi"
        }

        run_location_synthesis_experiment(
            vae_model, discriminators, attribute_names, test_ds,
            test_website_ids, df_original, source_features_3, "leuven", experiment_params)

        print("\n--- All Location Synthesis Experiments Complete ---")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
