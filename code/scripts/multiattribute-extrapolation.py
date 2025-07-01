import random
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

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

# Website identity model path
WEBSITE_MODEL_PATH = "../../models/website/Lausanne-Leuven-Singapore-baseCNN-online_semi_hard_AdamW-epochs1000-train_samples1200-batch128_best.keras"

# Base output directory for extrapolated traffic samples
BASE_OUTPUT_DIR = '../../figures/multi_attribute_extrapolation/'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Model parameters (should match training configuration)
SEQUENCE_LENGTH = 32
LATENT_DIM = 8

# Dataset feature definitions
LOCATIONS = ["lausanne", "leuven", "singapore"]
CLIENTS = ["cloudflare", "firefox"]
RESOLVERS = ["google", "cloudflare"]
PLATFORMS = ["desktop", "desktop_(aws)", "raspberry_pi"]

# Feature to attribute mapping
FEATURE_ATTRIBUTES = {
    "location": LOCATIONS,
    "client": CLIENTS,
    "resolver": RESOLVERS,
    "platform": PLATFORMS
}


class TripletTrainingConfig:
    """Configuration for triplet model"""

    def __init__(self, feature_length=32, num_train_samples=1200, batch_size=128,
                 epochs=1000, learning_rate=1e-4, weight_decay=1e-5, margin=0.2,
                 patience=50, validation_split=0.2, base_network_name='baseCNN'):
        self.feature_length = feature_length
        self.num_train_samples = num_train_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.margin = margin
        self.patience = patience
        self.validation_split = validation_split
        self.base_network_name = base_network_name


def create_base_cnn(input_length, embedding_dim=64):
    """Create base CNN for triplet model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_length,)),
        tf.keras.layers.Reshape((input_length, 1)),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])
    return model


def create_model_and_compile(base_network, config):
    """Create and compile triplet model"""
    # Create triplet inputs
    anchor_input = tf.keras.layers.Input(shape=(config.feature_length,))
    positive_input = tf.keras.layers.Input(shape=(config.feature_length,))
    negative_input = tf.keras.layers.Input(shape=(config.feature_length,))

    # Get embeddings
    anchor_embedding = base_network(anchor_input)
    positive_embedding = base_network(positive_input)
    negative_embedding = base_network(negative_input)

    # Create model
    model = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding]
    )

    return model


class MultiAttributeSynthesizer:
    """Enhanced synthesizer for multi-attribute extrapolation"""

    def __init__(self, vae_model, discriminators, website_model=None):
        self.vae_model = vae_model
        self.discriminators = discriminators
        self.website_model = website_model
        self.hyperplanes = {}

        # Pre-compute hyperplane parameters
        for attr_name, discriminator in discriminators.items():
            self.hyperplanes[attr_name] = Hyperplane(discriminator)

    def get_combined_direction_vector(self, target_changes: Dict[str, float]) -> tf.Tensor:
        """
        Compute weighted combination of direction vectors for multiple attributes.

        Args:
            target_changes: Dict mapping attribute names to their weights
                          (positive weights move towards the attribute)

        Returns:
            Combined direction vector
        """
        combined_vector = tf.zeros((1, LATENT_DIM))

        for attr_name, weight in target_changes.items():
            if attr_name in self.hyperplanes:
                direction_vector, _ = self.hyperplanes[attr_name].get_hyplerplane_params(
                )
                direction_vector = tf.expand_dims(direction_vector, axis=0)
                combined_vector += weight * direction_vector
                print(f"Added {attr_name} with weight {weight}")
            else:
                print(
                    f"Warning: No hyperplane found for attribute {attr_name}")

        # Normalize the combined vector to unit length
        combined_vector = tf.nn.l2_normalize(combined_vector, axis=1)
        return combined_vector

    def compute_identity_distance(self, original_sequence: tf.Tensor,
                                  synthesized_sequence: tf.Tensor) -> float:
        """
        Compute distance between original and synthesized sequences in triplet embedding space.
        """
        if self.website_model is None:
            return 0.0

        # Get embeddings for both sequences
        original_embedding = self.website_model.layers[3](
            original_sequence)  # base network
        synthesized_embedding = self.website_model.layers[3](
            synthesized_sequence)

        # Compute Euclidean distance
        distance = tf.norm(original_embedding - synthesized_embedding, axis=1)
        return distance.numpy()[0]

    def evaluate_discriminator_scores(self, latent_vector: tf.Tensor) -> Dict[str, float]:
        """Evaluate all discriminator scores for a given latent vector"""
        scores = {}
        for attr_name, discriminator in self.discriminators.items():
            score = discriminator(latent_vector).numpy()[0, 0]
            scores[attr_name] = score
        return scores

    def synthesize_multi_attribute(self,
                                   original_sequence: tf.Tensor,
                                   target_changes: Dict[str, float],
                                   experiment_params: Dict,
                                   preserve_attributes: List[str] = None) -> Tuple[List[tf.Tensor], Dict]:
        """
        Synthesize traffic by changing multiple attributes simultaneously.

        Args:
            original_sequence: Input traffic sequence [1, sequence_length]
            target_changes: Dict of attribute changes with weights
            experiment_params: Synthesis parameters
            preserve_attributes: List of attributes to keep fixed

        Returns:
            Tuple of (generated_sequences, synthesis_log)
        """
        preserve_attributes = preserve_attributes or []

        # Get combined direction vector
        direction_vector = self.get_combined_direction_vector(target_changes)

        # Encode original sequence
        initial_z_mean, _, _ = self.vae_model.encode(original_sequence)
        current_z = tf.identity(initial_z_mean)

        # Extract parameters
        num_steps = experiment_params.get('num_steps', 50)
        step_size = experiment_params.get('step_size', 0.1)
        pull_strength = experiment_params.get('pull_strength', 0.01)
        max_latent_norm = experiment_params.get(
            'max_latent_norm', 10 * np.sqrt(LATENT_DIM))
        identity_threshold = experiment_params.get('identity_threshold', 0.2)
        preserve_threshold = experiment_params.get('preserve_threshold', 0.3)

        generated_sequences = [original_sequence[0]]
        synthesis_log = {
            'discriminator_scores': [],
            'identity_distances': [],
            'latent_norms': [],
            'step_details': []
        }

        print(f"Starting multi-attribute synthesis...")
        print(f"Target changes: {target_changes}")
        print(f"Preserving attributes: {preserve_attributes}")

        for step in range(num_steps):
            # Get current scores
            current_scores = self.evaluate_discriminator_scores(current_z)
            current_norm = tf.norm(current_z).numpy()

            # Decode current latent vector
            decoded_sequence = self.vae_model.decode(current_z)

            # Compute identity distance if website model is available
            identity_distance = 0.0
            if self.website_model is not None:
                identity_distance = self.compute_identity_distance(
                    original_sequence, decoded_sequence)

            # Log current state
            synthesis_log['discriminator_scores'].append(current_scores.copy())
            synthesis_log['identity_distances'].append(identity_distance)
            synthesis_log['latent_norms'].append(current_norm)

            # Check stopping conditions
            stop_reason = None

            # 1. Check latent norm
            if current_norm > max_latent_norm:
                stop_reason = f"Latent norm exceeded threshold: {current_norm:.2f} > {max_latent_norm:.2f}"

            # 2. Check identity preservation
            elif self.website_model is not None and identity_distance > identity_threshold:
                stop_reason = f"Identity distance exceeded threshold: {identity_distance:.3f} > {identity_threshold:.3f}"

            # 3. Check preservation of specified attributes
            elif preserve_attributes:
                for attr in preserve_attributes:
                    if attr in current_scores and current_scores[attr] < preserve_threshold:
                        stop_reason = f"Preserved attribute '{attr}' score too low: {current_scores[attr]:.3f} < {preserve_threshold:.3f}"
                        break

            if stop_reason:
                print(f"Stopped at step {step + 1}: {stop_reason}")
                break

            # Apply synthesis step
            current_z = (1 - pull_strength) * current_z + \
                step_size * direction_vector

            # Decode and store
            decoded_sequence = self.vae_model.decode(current_z)
            generated_sequences.append(decoded_sequence[0])

            # Log step details
            step_detail = f"Step {step + 1}: "
            for attr, score in current_scores.items():
                step_detail += f"{attr}={score:.2f}, "
            step_detail += f"norm={current_norm:.2f}, id_dist={identity_distance:.3f}"

            synthesis_log['step_details'].append(step_detail)
            print(step_detail)

        print(
            f"Synthesis completed with {len(generated_sequences)} sequences.")
        return generated_sequences, synthesis_log


def select_samples_with_specific_features(df, feature_criteria, website_id, num_samples=1):
    """Select network traffic samples matching specified criteria"""
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

    # Extract features
    feature_cols = [str(i) for i in range(SEQUENCE_LENGTH)]
    selected_df = filtered_df.sample(n=num_samples, random_state=None)
    selected_sequences = selected_df[feature_cols].astype("float32").values

    return selected_sequences


def plot_multi_attribute_results(sequences, titles, target_changes, synthesis_log,
                                 experiment_name, output_dir):
    """Plot results of multi-attribute synthesis"""

    # 1. Plot sequence progression
    fig_cols = min(len(sequences), 4)
    fig_rows = int(np.ceil(len(sequences) / fig_cols))

    plt.figure(figsize=(5 * fig_cols, 4 * fig_rows))
    # Show max 8 sequences
    for i, seq in enumerate(sequences[::max(1, len(sequences)//8)]):
        plt.subplot(fig_rows, fig_cols, i + 1)
        seq_data = seq.numpy() if tf.is_tensor(seq) else seq
        plt.plot(seq_data, alpha=0.8, linewidth=1.2)
        plt.xlabel('Time Step')
        plt.ylabel('Packet Count')
        plt.grid(True, alpha=0.3)
        plt.title(f'Step {i * max(1, len(sequences)//8)}')

    target_str = ", ".join([f"{k}:{v:.1f}" for k, v in target_changes.items()])
    plt.suptitle(f'Multi-Attribute Synthesis: {target_str}', fontsize=12)
    plt.tight_layout()

    filename = f"{experiment_name}_sequence_progression.png"
    plt.savefig(os.path.join(output_dir, filename),
                dpi=150, bbox_inches='tight')
    print(f"Saved sequence progression plot to {filename}")
    plt.show()
    plt.close()

    # 2. Plot discriminator scores over time
    if synthesis_log['discriminator_scores']:
        attr_names = list(synthesis_log['discriminator_scores'][0].keys())

        plt.figure(figsize=(12, 8))

        # Plot each attribute score
        for i, attr in enumerate(attr_names):
            scores = [step_scores[attr]
                      for step_scores in synthesis_log['discriminator_scores']]
            plt.plot(scores, label=attr, alpha=0.8, linewidth=1.5)

        plt.xlabel('Synthesis Step')
        plt.ylabel('Discriminator Score')
        plt.title('Discriminator Scores During Multi-Attribute Synthesis')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{experiment_name}_discriminator_scores.png"
        plt.savefig(os.path.join(output_dir, filename),
                    dpi=150, bbox_inches='tight')
        print(f"Saved discriminator scores plot to {filename}")
        plt.show()
        plt.close()

    # 3. Plot identity distance and latent norm
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if synthesis_log['identity_distances'] and any(d > 0 for d in synthesis_log['identity_distances']):
        plt.plot(synthesis_log['identity_distances'], 'r-', linewidth=2)
        plt.xlabel('Synthesis Step')
        plt.ylabel('Identity Distance')
        plt.title('Website Identity Preservation')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.2, color='red', linestyle='--',
                    alpha=0.7, label='Threshold')
        plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(synthesis_log['latent_norms'], 'b-', linewidth=2)
    plt.xlabel('Synthesis Step')
    plt.ylabel('Latent Vector Norm')
    plt.title('Latent Space Distance from Origin')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"{experiment_name}_metrics.png"
    plt.savefig(os.path.join(output_dir, filename),
                dpi=150, bbox_inches='tight')
    print(f"Saved metrics plot to {filename}")
    plt.show()
    plt.close()


def load_models_and_data():
    """Load all necessary models and data"""
    print("Loading VAE model...")
    vae_model = model_utils.load_model(CHECKPOINT_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        raise FileNotFoundError('VAE model not found')
    print("VAE model loaded successfully.")

    print("Loading website identity model...")
    website_model = None
    if os.path.exists(WEBSITE_MODEL_PATH):
        try:
            config = TripletTrainingConfig()
            base_net = create_base_cnn(config.feature_length, embedding_dim=64)
            website_model = create_model_and_compile(base_net, config)
            website_model.load_weights(WEBSITE_MODEL_PATH)
            print("Website identity model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load website model: {e}")
    else:
        print("Website model not found. Identity preservation will be disabled.")

    print("Loading dataset...")
    train_ds, test_ds, train_website_ids, test_website_ids, attribute_names = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1, length=SEQUENCE_LENGTH)

    # Load original dataset
    df_original = pd.read_csv(DATASET_CSV, index_col=0)
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

    return vae_model, website_model, discriminators, attribute_names, test_website_ids, test_df


def run_multi_attribute_experiment(synthesizer, test_df, source_features,
                                   target_changes, preserve_attributes,
                                   experiment_params, website_id):
    """Run a multi-attribute synthesis experiment"""

    # Create experiment name and directory
    changes_str = "_".join([f"{k}{v:+.1f}" for k, v in target_changes.items()])
    experiment_name = f"multi_attr_{changes_str}_website_{website_id}"
    experiment_output_dir = os.path.join(BASE_OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_output_dir, exist_ok=True)

    print(f"\n--- Multi-Attribute Synthesis Experiment: {experiment_name} ---")

    # Select source sequence
    print(f"Selecting source sequence with features: {source_features}")
    try:
        source_sequences = select_samples_with_specific_features(
            test_df, source_features, website_id, num_samples=1
        )
        original_sequence = tf.expand_dims(source_sequences[0], axis=0)

    except ValueError as e:
        print(f"Error selecting source sample: {e}")
        return

    # Run synthesis
    generated_sequences, synthesis_log = synthesizer.synthesize_multi_attribute(
        original_sequence, target_changes, experiment_params, preserve_attributes
    )

    # Plot results
    step_titles = [f"Step {i}" for i in range(len(generated_sequences))]
    plot_multi_attribute_results(
        generated_sequences, step_titles, target_changes,
        synthesis_log, experiment_name, experiment_output_dir
    )

    print(f"--- Experiment Complete: {experiment_name} ---")

    return generated_sequences, synthesis_log


if __name__ == "__main__":
    print("--- Multi-Attribute Network Traffic Synthesis ---")

    # Load models and data
    vae_model, website_model, discriminators, attribute_names, test_website_ids, test_df = load_models_and_data()

    # Create synthesizer
    synthesizer = MultiAttributeSynthesizer(
        vae_model, discriminators, website_model)

    print(f"Available attributes: {attribute_names}")
    print(f"Available discriminators: {list(discriminators.keys())}")

    # Experiment parameters
    experiment_params = {
        'num_steps': 100,
        'step_size': 0.1,
        'pull_strength': 0.005,
        'max_latent_norm': 10 * np.sqrt(LATENT_DIM),
        'identity_threshold': 0.2,  # Triplet margin threshold
        'preserve_threshold': 0.3,
    }

    # Source features
    source_features = {
        "location": "leuven",
        "client": "cloudflare",
        "resolver": "cloudflare",
        "platform": "desktop"
    }

    # Example experiments
    experiments = [
        # Change location and client simultaneously
        {
            'target_changes': {
                'location_singapore': 1.0,
                'platform_desktop_(aws)': 0.8
            },
            'preserve_attributes': ['resolver_cloudflare', 'client_cloudflare']
        },

        # # Change location and resolver
        # {
        #     'target_changes': {
        #         'location_lausanne': 1.0,
        #         'resolver_google': 0.6
        #     },
        #     'preserve_attributes': ['client_cloudflare', 'platform_desktop']
        # },

        # # Change multiple attributes with different weights
        # {
        #     'target_changes': {
        #         'location_singapore': 1.2,
        #         'client_firefox': 0.5,
        #         'resolver_google': 0.3
        #     },
        #     'preserve_attributes': ['platform_desktop']
        # }
    ]

    # Run experiments
    for i, exp_config in enumerate(experiments):
        website_id = random.choice(test_website_ids)
        print(
            f"\n=== Running Experiment {i+1} with website_id: {website_id} ===")

        run_multi_attribute_experiment(
            synthesizer, test_df, source_features,
            exp_config['target_changes'],
            exp_config['preserve_attributes'],
            experiment_params, website_id
        )

    print("\n--- All Multi-Attribute Synthesis Experiments Complete ---")
