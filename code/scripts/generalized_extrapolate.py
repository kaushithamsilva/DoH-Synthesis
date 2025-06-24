import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Import your custom modules
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


def select_samples_with_specific_features(target_df, attr_names_list, feature_criteria):
    """
    Selects ALL network traffic samples from a DataFrame that match the specified feature criteria.
    Args:
        target_df (pd.DataFrame): The DataFrame to search (e.g., train_df or test_df).
        attr_names_list (list): List of all attribute names for index lookup.
        feature_criteria (dict): Dictionary with feature values.
                                 E.g., {"location": "lausanne", "client": "cloudflare"}
    Returns:
        tuple: (list of selected traffic sequences (np.array),
                list of selected attribute tensors (np.array),
                list of corresponding website IDs)
    """
    selected_sequences = []
    selected_attributes_full = []
    selected_website_ids = []

    # Convert feature criteria to binary attribute criteria for DataFrame filtering
    filter_conditions = pd.Series(True, index=target_df.index)

    # Track which attributes we actually use for filtering
    used_attribute_criteria = {}

    for feature, value in feature_criteria.items():
        if feature in FEATURE_ATTRIBUTES:
            found_feature_value = False
            for possible_value in FEATURE_ATTRIBUTES[feature]:
                attr_name = f"{feature}_{possible_value}"
                if attr_name in attr_names_list:
                    # Check if the attribute column exists in target_df
                    if attr_name in target_df.columns:
                        if possible_value == value:
                            filter_conditions &= (target_df[attr_name] == 1)
                            used_attribute_criteria[attr_name] = 1
                            found_feature_value = True
                        else:
                            filter_conditions &= (target_df[attr_name] == 0)
                            used_attribute_criteria[attr_name] = 0
                    else:
                        print(
                            f"Warning: Attribute column '{attr_name}' not found in DataFrame, skipping for filtering.")
                else:
                    print(
                        f"Warning: Attribute '{attr_name}' derived from '{feature}_{possible_value}' not in attribute_names_list.")
            if not found_feature_value:
                print(
                    f"Warning: Value '{value}' for feature '{feature}' not found in FEATURE_ATTRIBUTES mapping or corresponding attribute column.")
        else:
            print(
                f"Warning: Feature '{feature}' not defined in FEATURE_ATTRIBUTES mapping.")

    if not used_attribute_criteria:
        raise ValueError(
            "No valid attribute criteria found for sample selection based on feature_criteria. "
            "Please check FEATURE_ATTRIBUTES mapping and target_df columns."
        )

    # Apply all filter conditions
    matching_df = target_df[filter_conditions]

    if matching_df.empty:
        criteria_str = ", ".join(
            [f"{feature}={value}" for feature, value in feature_criteria.items()])
        raise ValueError(
            f"Could not find any traffic samples with features: {criteria_str}.")

    # Extract sequences, attributes, and website IDs from matching_df
    packet_cols = [str(i) for i in range(SEQUENCE_LENGTH)]
    for _, row in matching_df.iterrows():
        sequence = row[packet_cols].values.astype(float)
        # Extract attribute values directly from the row as a Series, then convert to numpy array
        attributes = row[attr_names_list].values.astype(int)
        # Assuming 'Website' column exists in your DataFrame
        website_id = row['Website']

        selected_sequences.append(sequence)
        selected_attributes_full.append(attributes)
        selected_website_ids.append(website_id)

    print(
        f"Found {len(selected_sequences)} traffic samples matching criteria: {feature_criteria}.")
    return selected_sequences, selected_attributes_full, selected_website_ids


def plot_traffic_sequences(sequences, titles=None, main_title="", filename="", output_dir=""):
    """
    Plots a list of network traffic sequences and saves them.
    """
    num_sequences = len(sequences)
    if num_sequences == 0:
        print("No sequences to plot.")
        return

    # Adjust cols and rows for better visualization of multiple sequences
    fig_cols = min(num_sequences, 5)  # Max 5 columns for reasonable layout
    fig_rows = int(np.ceil(num_sequences / fig_cols))

    # Adjust figsize for better readability
    plt.figure(figsize=(4 * fig_cols, 3 * fig_rows))
    for i, seq in enumerate(sequences):
        plt.subplot(fig_rows, fig_cols, i + 1)
        seq_data = seq.numpy() if tf.is_tensor(seq) else seq
        plt.plot(seq_data, alpha=0.8, linewidth=1.2)
        plt.xlabel('Time Step')
        plt.ylabel('Packet Count')
        plt.grid(True, alpha=0.3)
        if titles and i < len(titles):
            plt.title(titles[i], fontsize=10)  # Smaller title font
        else:
            plt.title(f'Step {i}', fontsize=10)

    plt.suptitle(main_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if filename:
        full_path = os.path.join(output_dir, filename)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {full_path}")
    plt.show()
    plt.close()


def plot_comparison_with_actual(actual_median_sequence, source_median_sequence, synthesized_median_sequence, title="", filename="", output_dir=""):
    """
    Plots comparison between actual, source, and final synthesized traffic median sequences.
    """
    plt.figure(figsize=(10, 6))  # Adjusted figsize for a single plot

    # Main comparison plot
    plt.plot(actual_median_sequence, label='Actual (Median)',
             alpha=0.8, linewidth=2, color='green')
    plt.plot(source_median_sequence, label='Source (Median)',
             alpha=0.8, linewidth=2, color='blue')
    plt.plot(synthesized_median_sequence, label='Synthesized (Median)',
             alpha=0.8, linewidth=2, color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Packet Count')
    plt.title(f'{title} - Median Traffic Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
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
    test_df,  # Now a DataFrame
    test_website_ids,  # Keep for context if needed, but not directly used for selection
    df_original,  # Original full DataFrame for actual comparisons
    source_features,
    target_location,
    experiment_params
):
    """
    Synthesizes traffic from source location to target location while keeping other features fixed.
    This now operates on ALL samples matching criteria and plots median traces.

    Args:
        vae_model (tf.keras.Model): The loaded VAE model.
        discriminators (dict): Dictionary of loaded discriminator models.
        attribute_names (list): List of all attribute names.
        test_df (pd.DataFrame): The test DataFrame for sample selection.
        test_website_ids (list): List of website IDs corresponding to test samples (for general context).
        df_original (pd.DataFrame): Original dataset DataFrame for getting actual traffic sequences.
        source_features (dict): Source feature configuration.
        target_location (str): Target location to synthesize.
        experiment_params (dict): Experiment configuration parameters.
    """
    print(
        f"\n--- Starting Location Synthesis: {source_features['location']} → {target_location} ---")

    # Set up experiment-specific output directory
    experiment_name = f"{source_features['location']}_to_{target_location}"
    # Append attributes for more specific naming if multiple runs with same location change
    suffix_parts = [f"{k}-{v}" for k,
                    v in source_features.items() if k != "location"]
    experiment_name += "_" + "_".join(suffix_parts) if suffix_parts else ""

    experiment_output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Select ALL initial traffic samples based on source features from the test_df
    print(
        f"Selecting all traffic samples with source features: {source_features} from test_df...")
    try:
        initial_sequences_batch, initial_attrs_batch, initial_website_ids = select_samples_with_specific_features(
            test_df,
            attribute_names,
            source_features
        )
        # Convert to TensorFlow tensor for batch processing
        original_sequences_tf = tf.stack(initial_sequences_batch)
        print(f"Selected {original_sequences_tf.shape[0]} initial sequences.")

        # Calculate and plot median of source sequences
        source_median_sequence = np.median(initial_sequences_batch, axis=0)
        plot_traffic_sequences([source_median_sequence],
                               titles=[
                                   f"Source Traffic Median (from {original_sequences_tf.shape[0]} samples)"],
                               main_title=f"Source Median: {source_features}",
                               filename="source_median_traffic.png",
                               output_dir=experiment_output_dir)

    except ValueError as e:
        print(f"Error selecting initial samples: {e}")
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

    # Fixed feature discriminators (for stabilization)
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
    direction_vector = tf.expand_dims(
        direction_vector, axis=0)  # Make it (1, LATENT_DIM)

    print(
        f"Direction vector shape for {target_location}: {direction_vector.shape}")

    # Encode the original traffic sequences (now a batch)
    initial_z_mean, _, _ = vae_model.encode(original_sequences_tf)
    # current_z is now (batch_size, LATENT_DIM)
    current_z = tf.identity(initial_z_mean)

    # Extrapolation parameters
    num_steps = experiment_params.get('num_steps', 25)
    step_size = experiment_params.get('step_size', 0.2)
    pull_strength = experiment_params.get(
        'pull_strength', 0.015)  # Pull towards origin
    target_threshold = experiment_params.get('target_threshold', 5.0)
    fixed_threshold = experiment_params.get('fixed_threshold', 1.0)

    max_latent_norm_threshold = 10 * np.sqrt(LATENT_DIM)
    print(f"Maximum allowed latent norm: {max_latent_norm_threshold:.2f}")
    print(f"Pull-to-center strength (beta): {pull_strength}")
    print(f"Target location threshold: {target_threshold}")
    print(f"Fixed features threshold: {fixed_threshold}")

    generated_median_sequences = [source_median_sequence]
    step_titles_progression = [f"Step 0 (Source Median)"]

    print(
        f"\nStarting synthesis from {source_features['location']} to {target_location} (batch size {current_z.shape[0]})...")

    for i in range(num_steps):
        # Get current discriminator scores for the entire batch
        target_scores = target_location_discriminator(
            current_z).numpy().flatten()

        # Calculate mean scores for fixed features across the batch
        fixed_mean_scores = {}
        for attr_name, disc in fixed_discriminators.items():
            fixed_mean_scores[attr_name] = np.mean(
                disc(current_z).numpy().flatten())

        current_latent_norm = np.mean(
            tf.norm(current_z, axis=1).numpy())  # Mean norm of the batch

        # Stop condition 1: Median target location threshold reached
        # We check the median score for the target, not individual scores
        if np.median(target_scores) > target_threshold:
            print(
                f"Stopped at step {i+1}: Median target location '{target_location}' threshold reached. Median Score: {np.median(target_scores):.2f} > {target_threshold:.2f}")
            break

        # Stop condition 2: Latent vector too far from origin (mean norm)
        if current_latent_norm > max_latent_norm_threshold:
            print(
                f"Stopped at step {i+1}: Mean Latent vector norm ({current_latent_norm:.2f}) exceeded threshold.")
            break

        # Stop condition 3: Any fixed feature deviates too much (mean score)
        should_stop_fixed_feature = False
        for attr_name, mean_score in fixed_mean_scores.items():
            # For fixed features, we assume they should ideally stay positive (score > 0)
            # or at least above a certain threshold (e.g., 0.5 or 1.0 depending on discriminator output)
            if mean_score < fixed_threshold:
                print(
                    f"Stopped at step {i+1}: Fixed feature '{attr_name}' median deviated. Mean Score: {mean_score:.2f} < {fixed_threshold:.2f}")
                should_stop_fixed_feature = True
                break
        if should_stop_fixed_feature:
            break

        # Apply extrapolation step with pull-to-center ONLY
        # Removed other_location_aversion_strength and other_fixed_aversion_strength
        current_z = (1 - pull_strength) * current_z + \
            step_size * direction_vector

        # Decode the new latent vectors
        decoded_sequences_batch = vae_model.decode(
            current_z)  # (batch_size, SEQUENCE_LENGTH)

        # Calculate median and append for plotting
        median_decoded_seq = np.median(decoded_sequences_batch.numpy(), axis=0)
        generated_median_sequences.append(median_decoded_seq)
        step_titles_progression.append(f"Step {i+1}")

        # Log progress
        log_str = f"Step {i+1}: {target_location} Median = {np.median(target_scores):.2f}"
        for attr_name, score in fixed_mean_scores.items():
            log_str += f", {attr_name} Mean = {score:.2f}"
        log_str += f", Mean Norm = {current_latent_norm:.2f}"
        print(log_str)

    print(
        f"Synthesis completed. Generated {len(generated_median_sequences)} median traffic sequences.")

    # Visualize progression of median sequences
    plot_traffic_sequences(generated_median_sequences,
                           titles=step_titles_progression,
                           main_title=f"Location Synthesis Progression (Median): {source_features['location']} → {target_location}",
                           filename=f"{experiment_name}_median_progression.png",
                           output_dir=experiment_output_dir)

    # Prepare for final comparison plot
    final_synthesized_median_sequence = generated_median_sequences[-1]

    # Get actual target location traffic median for comparison
    target_features = source_features.copy()
    target_features['location'] = target_location

    try:
        actual_target_sequences, _, _ = select_samples_with_specific_features(
            df_original,  # Use df_original to get actual samples
            attribute_names,
            target_features
        )
        actual_target_median_sequence = np.median(
            actual_target_sequences, axis=0)

        # Plot comparison: actual target median, source median, synthesized median
        plot_comparison_with_actual(actual_target_median_sequence,
                                    source_median_sequence,  # Already calculated
                                    final_synthesized_median_sequence,
                                    title=f"Synthesis Comparison: {source_features['location']} → {target_location}",
                                    filename=f"{experiment_name}_median_comparison.png",
                                    output_dir=experiment_output_dir)

    except ValueError as e:
        print(
            f"Could not find actual {target_location} traffic with same features for comparison (Error: {e}). "
            f"Plotting source median vs synthesized median only."
        )
        # If target actual not found, compare source median with synthesized median
        # For simplicity, we can still use the same comparison function, just pass source_median as actual
        plot_comparison_with_actual(source_median_sequence,  # Using source median as "actual" for this plot
                                    source_median_sequence,
                                    final_synthesized_median_sequence,
                                    title=f"Synthesis Comparison (No Actual Target Found): {source_features['location']} → {target_location}",
                                    filename=f"{experiment_name}_no_actual_target_comparison.png",
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
    # Load original full DataFrame
    df_original = pd.read_csv(DATASET_CSV, index_col=0)

    # Get all attribute names from the original DataFrame's columns, excluding traffic data and 'Website'
    packet_cols = [str(i) for i in range(SEQUENCE_LENGTH)]
    attribute_names = [
        col for col in df_original.columns if col not in packet_cols and col != 'Website']

    # Get train/test website IDs and split dataset into train_df and test_df
    # We still use get_train_test_dataset to get the website IDs split, but we'll use df_original to make DataFrames
    _, _, train_website_ids, test_website_ids, _ = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1)  # batch_size doesn't matter here as we use df_original

    train_df = df_original[df_original['Website'].isin(
        train_website_ids)].copy()
    test_df = df_original[df_original['Website'].isin(test_website_ids)].copy()

    print(f"Original dataset shape: {df_original.shape}")
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")

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

    return vae_model, discriminators, attribute_names, test_df, test_website_ids, df_original


if __name__ == "__main__":
    print("--- Initializing Network Traffic Location Synthesis Script ---")

    # Load all models and data
    vae_model, discriminators, attribute_names, test_df, test_website_ids, df_original = load_models_and_data()

    print(f"Available attributes: {attribute_names}")
    print(f"Available discriminators: {list(discriminators.keys())}")

    # Experiment parameters
    experiment_params = {
        'num_steps': 50,  # Increased steps for smoother progression
        'step_size': 0.1,
        'pull_strength': 0.015,  # Pull towards origin
        'target_threshold': 5.0,  # Stop when target location score > threshold
        'fixed_threshold': 1.0,   # Stop if any fixed feature score < threshold
    }

    # --- Example Experiments ---

    # Example 1: Lausanne (Cloudflare, desktop) -> Leuven
    source_features_1 = {
        "location": "lausanne",
        "client": "cloudflare",
        "resolver": "cloudflare",
        "platform": "desktop"
    }
    run_location_synthesis_experiment(
        vae_model, discriminators, attribute_names, test_df,
        test_website_ids, df_original, source_features_1, "leuven", experiment_params)

    # # Example 2: Lausanne (Firefox, desktop) -> Singapore
    # source_features_2 = {
    #     "location": "lausanne",
    #     "client": "firefox",
    #     "resolver": "google",
    #     "platform": "desktop"
    # }
    # run_location_synthesis_experiment(
    #     vae_model, discriminators, attribute_names, test_df,
    #     test_website_ids, df_original, source_features_2, "singapore", experiment_params)

    # # Example 3: Singapore (Cloudflare, Raspberry Pi) -> Leuven
    # source_features_3 = {
    #     "location": "singapore",
    #     "client": "cloudflare",
    #     "resolver": "google",
    #     "platform": "raspberry_pi"
    # }
    # run_location_synthesis_experiment(
    #     vae_model, discriminators, attribute_names, test_df,
    #     test_website_ids, df_original, source_features_3, "leuven", experiment_params)

    print("\n--- All Location Synthesis Experiments Complete ---")
