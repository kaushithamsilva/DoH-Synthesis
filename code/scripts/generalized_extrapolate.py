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
LATENT_DIM = 64  # Adjust based on your VAE configuration


def get_attribute_index(attribute_name, attr_names_list):
    """
    Helper to get the index of a specific attribute.
    """
    try:
        return attr_names_list.index(attribute_name)
    except ValueError:
        raise ValueError(
            f"Attribute '{attribute_name}' not found in attribute list. Available: {attr_names_list}")


def select_samples_with_specific_attributes(dataset, website_ids, attr_names_list, attribute_criteria, num_samples=1):
    """
    Selects a specified number of network traffic samples that match ALL desired attribute criteria.
    Args:
        dataset (tf.data.Dataset): The dataset to search.
        website_ids (list): List of website IDs corresponding to dataset samples.
        attr_names_list (list): List of all attribute names for index lookup.
        attribute_criteria (dict): A dictionary where keys are attribute names (str)
                                   and values are their desired binary states (0 or 1).
                                   E.g., {"location_lausanne": 1, "resolver_cloudflare": 1}
        num_samples (int): Number of samples to find.
    Returns:
        tuple: (list of selected traffic sequences, list of selected attribute tensors, list of website IDs)
    """
    selected_sequences = []
    selected_attributes_full = []
    selected_website_ids = []

    # Prepare attribute indices and desired values from criteria
    attribute_indices_to_check = []
    desired_values_for_check = []
    for attr_name, attr_value in attribute_criteria.items():
        attr_idx = get_attribute_index(attr_name, attr_names_list)
        attribute_indices_to_check.append(attr_idx)
        desired_values_for_check.append(int(attr_value))

    if not attribute_indices_to_check:
        raise ValueError(
            "No attribute criteria provided for sample selection.")

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
            [f"{name}={val}" for name, val in attribute_criteria.items()])
        raise ValueError(
            f"Could not find {num_samples} traffic samples with attributes: {criteria_str}.")

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


def run_traffic_extrapolation_experiment(
    vae_model,
    discriminators,
    attribute_names,
    test_ds,
    test_website_ids,
    df_original,
    experiment_params
):
    """
    Runs a generalized latent space extrapolation experiment for network traffic data.

    Args:
        vae_model (tf.keras.Model): The loaded VAE model.
        discriminators (dict): Dictionary of loaded discriminator models.
        attribute_names (list): List of all attribute names.
        test_ds (tf.data.Dataset): The test dataset for sample selection.
        test_website_ids (list): List of website IDs corresponding to test samples.
        df_original (pd.DataFrame): Original dataset for getting actual traffic sequences.
        experiment_params (dict): A dictionary containing experiment configuration.
    """
    print(
        f"\n--- Starting Traffic Experiment: {experiment_params['title']} ---")

    # Set up experiment-specific output directory
    experiment_output_dir = os.path.join(
        BASE_OUTPUT_DIR, experiment_params['output_filename_suffix'])
    os.makedirs(experiment_output_dir, exist_ok=True)

    # Get attribute indices
    try:
        attr_to_change_idx = get_attribute_index(
            experiment_params['attribute_to_change'], attribute_names)

        fixed_attr_idx = None
        if experiment_params.get('fixed_attribute'):
            fixed_attr_idx = get_attribute_index(
                experiment_params['fixed_attribute'], attribute_names)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Select initial traffic sample based on criteria
    print(
        f"Selecting traffic sample with criteria: {experiment_params['initial_sample_criteria']}...")
    try:
        initial_sequences, _, website_ids = select_samples_with_specific_attributes(
            test_ds,
            test_website_ids,
            attribute_names,
            experiment_params['initial_sample_criteria'],
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
                                   f"Original Traffic (Website {website_id})"],
                               main_title=f"Initial Sample: {experiment_params['title']}",
                               filename="original_traffic.png",
                               output_dir=experiment_output_dir)

    except ValueError as e:
        print(f"Error selecting initial sample: {e}")
        return

    # Load relevant discriminators
    print("Loading attribute discriminators...")
    attr_to_change_discriminator = discriminators.get(
        experiment_params['attribute_to_change'])
    fixed_discriminator = None

    if experiment_params.get('fixed_attribute'):
        fixed_discriminator = discriminators.get(
            experiment_params['fixed_attribute'])

    if not attr_to_change_discriminator:
        print(
            f"Failed to load discriminator for {experiment_params['attribute_to_change']}. Exiting experiment.")
        return

    if experiment_params.get('fixed_attribute') and not fixed_discriminator:
        print(
            f"Failed to load discriminator for {experiment_params['fixed_attribute']}. Exiting experiment.")
        return

    print("Discriminator models loaded successfully.")

    # Extract hyperplane parameters for the changing attribute
    attr_to_change_hyperplane = Hyperplane(attr_to_change_discriminator)
    base_direction_vector, _ = attr_to_change_hyperplane.get_hyplerplane_params()

    # Determine the actual direction for extrapolation
    if experiment_params['change_direction_towards_positive']:
        direction_vector = tf.expand_dims(base_direction_vector, axis=0)
    else:
        direction_vector = -tf.expand_dims(base_direction_vector, axis=0)

    print(
        f"Extrapolation direction vector (for {experiment_params['attribute_to_change']}) shape: {direction_vector.shape}")

    # Encode the original traffic sequence
    initial_z_mean, _, _ = vae_model.encode(original_sequence)
    current_z = tf.identity(initial_z_mean)

    # Extrapolate with pull-to-center
    num_extrapolation_steps = experiment_params['num_extrapolation_steps']
    step_size = experiment_params['step_size']
    pull_strength = experiment_params['pull_strength']

    max_latent_norm_threshold = 10 * np.sqrt(LATENT_DIM)
    print(f"Maximum allowed latent norm: {max_latent_norm_threshold:.2f}")
    print(f"Pull-to-center strength (beta): {pull_strength}")

    generated_sequences = [original_sequence[0]]
    step_titles = ["Original"]

    print("\nStarting extrapolation...")
    for i in range(num_extrapolation_steps):
        attr_to_change_score_current = attr_to_change_discriminator(current_z).numpy()[
            0, 0]
        fixed_attr_score_current = None
        if fixed_discriminator:
            fixed_attr_score_current = fixed_discriminator(current_z).numpy()[
                0, 0]

        current_latent_norm = tf.norm(current_z).numpy()

        # Stop conditions
        # Condition 1: Target attribute state reached
        if (experiment_params['change_direction_towards_positive'] and
            attr_to_change_score_current > experiment_params['target_attr_stop_threshold']) or \
           (not experiment_params['change_direction_towards_positive'] and
                attr_to_change_score_current < experiment_params['target_attr_stop_threshold']):
            print(
                f"Stopped at step {i+1}: Target attribute '{experiment_params['attribute_to_change']}' state reached. Score: {attr_to_change_score_current:.2f}")
            break

        # Condition 2: Latent vector too far from origin
        if current_latent_norm > max_latent_norm_threshold:
            print(
                f"Stopped at step {i+1}: Latent vector norm ({current_latent_norm:.2f}) exceeded threshold ({max_latent_norm_threshold:.2f}).")
            break

        # Condition 3: Fixed attribute not preserved
        if fixed_discriminator and experiment_params.get('fixed_attribute'):
            fixed_attr_name = experiment_params['fixed_attribute']
            stability_threshold = experiment_params['fixed_attr_stability_threshold']

            desired_fixed_attr_initial_val = experiment_params['initial_sample_criteria'].get(
                fixed_attr_name, 1)

            should_stop = False
            stop_reason = ""

            if desired_fixed_attr_initial_val == 1:
                if fixed_attr_score_current < stability_threshold:
                    should_stop = True
                    stop_reason = f"Fixed attribute '{fixed_attr_name}' (desired positive) not preserved. Score: {fixed_attr_score_current:.2f} < Threshold: {stability_threshold:.2f}"
            else:
                if fixed_attr_score_current > stability_threshold:
                    should_stop = True
                    stop_reason = f"Fixed attribute '{fixed_attr_name}' (desired negative) not preserved. Score: {fixed_attr_score_current:.2f} > Threshold: {stability_threshold:.2f}"

            if should_stop:
                print(f"Stopped at step {i+1}: {stop_reason}")
                break

        # Apply the pull-to-center logic and move along the direction vector
        current_z = (1 - pull_strength) * current_z + \
            step_size * direction_vector

        # Decode the new latent vector
        decoded_sequence = vae_model.decode(current_z)
        generated_sequences.append(decoded_sequence[0])
        step_titles.append(f"Step {i+1}")

        # Log scores
        log_str = f"Step {i+1}: {experiment_params['attribute_to_change']} Score = {attr_to_change_discriminator(current_z).numpy()[0,0]:.2f}"
        if fixed_discriminator:
            log_str += f", {experiment_params['fixed_attribute']} Score = {fixed_discriminator(current_z).numpy()[0,0]:.2f}"
        log_str += f", Latent Norm = {tf.norm(current_z).numpy():.2f}"
        print(log_str)

    print(
        f"Extrapolation completed. Generated {len(generated_sequences)} traffic sequences.")

    # Visualize results
    plot_traffic_sequences(generated_sequences,
                           titles=step_titles,
                           main_title=f"{experiment_params['title']} (Epoch {CHECKPOINT_EPOCH}, Beta={pull_strength})",
                           filename=f"{experiment_params['output_filename_suffix']}_progression.png",
                           output_dir=experiment_output_dir)

    # Plot comparison with actual traffic
    plot_comparison_with_actual(actual_sequence,
                                generated_sequences[0].numpy(),
                                generated_sequences[-1].numpy(),
                                title=f"{experiment_params['title']} - Website {website_id}",
                                filename=f"{experiment_params['output_filename_suffix']}_comparison.png",
                                output_dir=experiment_output_dir)

    print(
        f"--- Traffic Experiment '{experiment_params['title']}' Complete ---")


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
    print("--- Initializing Network Traffic Extrapolation Script ---")

    try:
        # Load all models and data
        vae_model, discriminators, attribute_names, test_ds, test_website_ids, df_original = load_models_and_data()

        print(f"Available attributes: {attribute_names}")
        print(f"Available discriminators: {list(discriminators.keys())}")

        # Example experiment: Change location while keeping resolver fixed
        location_change_experiment = {
            "initial_sample_criteria": {"location_lausanne": 1, "resolver_cloudflare": 1},
            "attribute_to_change": "location_lausanne",
            "change_direction_towards_positive": False,  # From Lausanne to not-Lausanne
            # Stop when location_lausanne score becomes negative
            "target_attr_stop_threshold": -0.5,
            "fixed_attribute": "resolver_cloudflare",
            "fixed_attr_stability_threshold": 0.5,  # Keep resolver_cloudflare positive
            "num_extrapolation_steps": 25,
            "step_size": 0.6,
            "pull_strength": 0.015,
            "output_filename_suffix": "location_change_keep_resolver",
            "title": "Network Traffic: Change Location, Keep Resolver"
        }

        # Run the experiment
        run_traffic_extrapolation_experiment(
            vae_model, discriminators, attribute_names, test_ds,
            test_website_ids, df_original, location_change_experiment)

        # # Example experiment 2: Change resolver while keeping location fixed
        # resolver_change_experiment = {
        #     "initial_sample_criteria": {"location_lausanne": 1, "resolver_cloudflare": 1},
        #     "attribute_to_change": "resolver_cloudflare",
        #     "change_direction_towards_positive": False,  # From Cloudflare to not-Cloudflare
        #     "target_attr_stop_threshold": -0.5,
        #     "fixed_attribute": "location_lausanne",
        #     "fixed_attr_stability_threshold": 0.5,
        #     "num_extrapolation_steps": 25,
        #     "step_size": 0.2,
        #     "pull_strength": 0.015,
        #     "output_filename_suffix": "resolver_change_keep_location",
        #     "title": "Network Traffic: Change Resolver, Keep Location"
        # }

        # run_traffic_extrapolation_experiment(
        #     vae_model, discriminators, attribute_names, test_ds,
        #     test_website_ids, df_original, resolver_change_experiment)

        print("\n--- All Traffic Extrapolation Experiments Complete ---")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
