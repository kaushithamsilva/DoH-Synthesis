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
CHECKPOINT_EPOCH = 950
DISCRIMINATOR_PATH = CHECKPOINT_PATH
VAE_MODEL_NAME = f'vae_e{CHECKPOINT_EPOCH}'

# Base output directory for synthesized datasets
BASE_OUTPUT_DIR = '../../dataset/synthesized/'
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

# Regularization strength for pulling towards center
CENTER_PULL_REGULARIZER = 0.001


class BatchMultiAttributeSynthesizer:
    """Efficient batch synthesizer using direct hyperplane projection"""

    def __init__(self, vae_model, discriminators):
        self.vae_model = vae_model
        self.discriminators = discriminators
        self.hyperplanes = {}

        # Pre-compute hyperplane parameters
        for attr_name, discriminator in discriminators.items():
            self.hyperplanes[attr_name] = Hyperplane(discriminator)
            print(f"Loaded hyperplane for {attr_name}")

    def project_to_hyperplane_with_offset(self, z_batch: tf.Tensor,
                                          target_changes: Dict[str, float],
                                          alpha: float = 1.0) -> tf.Tensor:
        """
        Project latent vectors to hyperplane with offset using direct mathematical approach.

        Mathematical approach:
        1. For each hyperplane defined by w^T * z + b = 0 (where w is normal vector, b is bias)
        2. Distance from point z to hyperplane: d = (w^T * z + b) / ||w||
        3. Projection of z onto hyperplane: z_proj = z - d * (w / ||w||)
        4. Move alpha distance away from boundary: z_final = z_proj + alpha * (w / ||w||)

        For multiple attributes:
        - Compute weighted combination of normal vectors
        - Project to combined hyperplane surface

        Args:
            z_batch: Batch of latent vectors [batch_size, latent_dim]
            target_changes: Dict mapping attribute names to their weights
            alpha: Distance to move away from hyperplane boundary

        Returns:
            Projected latent vectors [batch_size, latent_dim]
        """
        batch_size = tf.shape(z_batch)[0]

        # Compute combined normal vector (weighted sum of hyperplane normals)
        combined_normal = tf.zeros((LATENT_DIM,))
        combined_bias = 0.0
        total_weight = 0.0

        for attr_name, weight in target_changes.items():
            if attr_name in self.hyperplanes:
                # Get hyperplane parameters: w (normal vector) and b (bias)
                normal_vector, bias = self.hyperplanes[attr_name].get_hyplerplane_params(
                )

                # Add weighted contribution
                combined_normal += weight * normal_vector
                combined_bias += weight * bias
                total_weight += abs(weight)

        if total_weight == 0:
            print("Warning: No valid hyperplanes found for target changes")
            return z_batch

        # Normalize combined normal vector
        combined_normal = tf.nn.l2_normalize(combined_normal, axis=0)
        combined_bias = combined_bias / total_weight

        # Vectorized projection for entire batch
        # Distance from each point to combined hyperplane: d = (w^T * z + b) / ||w||
        # Since w is normalized, ||w|| = 1
        distances = tf.reduce_sum(
            z_batch * combined_normal, axis=1) + combined_bias  # [batch_size,]
        distances = tf.expand_dims(distances, axis=1)  # [batch_size, 1]

        # Project to hyperplane: z_proj = z - d * w
        z_projected = z_batch - distances * \
            tf.expand_dims(combined_normal, axis=0)

        # Move alpha distance away from boundary in positive normal direction
        z_final = z_projected + alpha * tf.expand_dims(combined_normal, axis=0)

        # regularization: move towards the center of the latent space
        z_center = tf.zeros_like(z_final)
        z_final = z_final + CENTER_PULL_REGULARIZER * (z_center - z_final)

        return z_final

    def batch_synthesize(self, source_sequences: tf.Tensor,
                         target_changes: Dict[str, float],
                         alpha: float = 1.0,
                         preserve_attributes: List[str] = None,
                         preserve_threshold: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor, np.ndarray]:
        """
        Synthesize multiple sequences in batch using direct hyperplane projection.

        Args:
            source_sequences: Input sequences [batch_size, sequence_length]
            target_changes: Dict of attribute changes with weights
            alpha: Distance to move from hyperplane boundary
            preserve_attributes: List of attributes to preserve
            preserve_threshold: Minimum score threshold for preserved attributes

        Returns:
            Tuple of (synthesized_sequences, valid_mask, discriminator_scores)
        """
        preserve_attributes = preserve_attributes or []

        print(
            f"Starting batch synthesis for {tf.shape(source_sequences)[0]} sequences...")
        print(f"Target changes: {target_changes}")
        print(f"Alpha (boundary distance): {alpha}")

        # Encode source sequences to latent space
        z_mean, _, _ = self.vae_model.encode(source_sequences)

        # Project to hyperplane with offset
        z_synthesized = self.project_to_hyperplane_with_offset(
            z_mean, target_changes, alpha
        )

        # Decode synthesized latent vectors
        synthesized_sequences = self.vae_model.decode(z_synthesized)

        # Evaluate all discriminators on synthesized latent vectors
        all_scores = {}
        for attr_name, discriminator in self.discriminators.items():
            scores = discriminator(z_synthesized)  # [batch_size, 1]
            all_scores[attr_name] = tf.squeeze(scores, axis=1)  # [batch_size]

        # Create validity mask based on preservation requirements
        valid_mask = tf.ones(tf.shape(source_sequences)[0], dtype=tf.bool)

        for preserve_attr in preserve_attributes:
            if preserve_attr in all_scores:
                attr_valid = all_scores[preserve_attr] >= preserve_threshold
                valid_mask = tf.logical_and(valid_mask, attr_valid)
                valid_count = tf.reduce_sum(tf.cast(attr_valid, tf.int32))
                print(
                    f"Attribute {preserve_attr}: {valid_count}/{tf.shape(source_sequences)[0]} samples above threshold {preserve_threshold}")

        total_valid = tf.reduce_sum(tf.cast(valid_mask, tf.int32))
        print(
            f"Total valid samples after filtering: {total_valid}/{tf.shape(source_sequences)[0]}")

        # Convert scores to numpy for easier handling
        scores_np = {attr: scores.numpy()
                     for attr, scores in all_scores.items()}

        return synthesized_sequences, valid_mask, scores_np

    def evaluate_synthesis_quality(self, original_sequences: tf.Tensor,
                                   synthesized_sequences: tf.Tensor,
                                   target_changes: Dict[str, float]) -> Dict:
        """Evaluate the quality of synthesized sequences"""

        # Encode both original and synthesized
        z_orig, _, _ = self.vae_model.encode(original_sequences)
        z_synth, _, _ = self.vae_model.encode(synthesized_sequences)

        # Compute latent space distance
        latent_distances = tf.norm(z_synth - z_orig, axis=1)

        # Evaluate discriminators on both
        orig_scores = {}
        synth_scores = {}

        for attr_name, discriminator in self.discriminators.items():
            orig_scores[attr_name] = tf.squeeze(discriminator(z_orig), axis=1)
            synth_scores[attr_name] = tf.squeeze(
                discriminator(z_synth), axis=1)

        # Compute score changes
        score_changes = {}
        for attr_name in orig_scores:
            score_changes[attr_name] = synth_scores[attr_name] - \
                orig_scores[attr_name]

        quality_metrics = {
            'latent_distances': latent_distances.numpy(),
            'original_scores': {k: v.numpy() for k, v in orig_scores.items()},
            'synthesized_scores': {k: v.numpy() for k, v in synth_scores.items()},
            'score_changes': {k: v.numpy() for k, v in score_changes.items()}
        }

        return quality_metrics


def load_models_and_data():
    """Load all necessary models and data"""
    print("Loading VAE model...")
    vae_model = model_utils.load_model(CHECKPOINT_PATH, VAE_MODEL_NAME)
    if vae_model is None:
        raise FileNotFoundError('VAE model not found')
    print("VAE model loaded successfully.")

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

    return vae_model, discriminators, attribute_names, test_website_ids, test_df


def select_source_samples(df, source_criteria, website_ids=None):
    """Select all source samples matching criteria for specified website IDs"""

    # Filter by website IDs if provided
    if website_ids is not None:
        df = df[df['Website'].isin(website_ids)].copy()

    # Build mask for source criteria
    mask = np.ones(len(df), dtype=bool)
    for feature, value in source_criteria.items():
        mask &= (df[feature.capitalize()] == value.capitalize())

    filtered_df = df[mask]

    print(
        f"Selected {len(filtered_df)} samples matching criteria across {len(filtered_df['Website'].unique())} websites")

    # Extract sequence features
    feature_cols = [str(i) for i in range(SEQUENCE_LENGTH)]
    sequences = filtered_df[feature_cols].astype("float32").values

    # Extract metadata
    metadata = filtered_df[['Website', 'Location',
                            'Resolver', 'Client', 'Platform']].copy()

    return sequences, metadata, filtered_df


def create_synthesized_metadata(original_metadata: pd.DataFrame,
                                target_changes: Dict[str, float],
                                valid_indices: np.ndarray) -> pd.DataFrame:
    """Create metadata for synthesized samples with updated attributes"""

    # Filter to valid samples only
    synth_metadata = original_metadata.iloc[valid_indices].copy(
    ).reset_index(drop=True)

    # Update attributes based on target changes
    for attr_change in target_changes.keys():
        # Parse attribute name (e.g., 'location_singapore' -> 'location', 'singapore')
        if '_' in attr_change:
            attr_category, attr_value = attr_change.split('_', 1)

            # Map to DataFrame column name
            column_map = {
                'location': 'Location',
                'client': 'Client',
                'resolver': 'Resolver',
                'platform': 'Platform'
            }

            if attr_category in column_map:
                column_name = column_map[attr_category]
                synth_metadata[column_name] = attr_value.capitalize()
                print(
                    f"Updated {column_name} to {attr_value.capitalize()} for {len(synth_metadata)} samples")

    return synth_metadata


def save_synthesized_dataset(sequences: np.ndarray,
                             metadata: pd.DataFrame,
                             target_changes: Dict[str, float],
                             quality_metrics: Dict = None):
    """Save synthesized dataset to CSV"""

    # Create filename based on target changes
    target_str = "_".join(
        [f"{k.replace('_', '-')}" for k in target_changes.keys()])
    filename = f"synthesized_{target_str}.csv"
    filepath = os.path.join(BASE_OUTPUT_DIR, filename)

    # Create DataFrame with metadata and sequence features
    result_df = metadata.copy()

    # Add sequence features as columns
    feature_cols = [str(i) for i in range(SEQUENCE_LENGTH)]
    for i, col in enumerate(feature_cols):
        result_df[col] = sequences[:, i]

    # Add synthesis metadata
    result_df['synthesis_target'] = str(target_changes)

    # Save to CSV
    result_df.to_csv(filepath, index=False)
    print(
        f"Saved synthesized dataset with {len(result_df)} samples to: {filepath}")

    # Save quality metrics if provided
    if quality_metrics:
        metrics_file = filepath.replace('.csv', '_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Synthesis Quality Metrics\n")
            f.write(f"Target Changes: {target_changes}\n\n")

            for metric_name, values in quality_metrics.items():
                if isinstance(values, dict):
                    f.write(f"{metric_name}:\n")
                    for sub_name, sub_values in values.items():
                        f.write(
                            f"  {sub_name}: mean={np.mean(sub_values):.4f}, std={np.std(sub_values):.4f}\n")
                else:
                    f.write(
                        f"{metric_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}\n")

        print(f"Saved quality metrics to: {metrics_file}")

    return filepath


def run_batch_synthesis_experiment(synthesizer, test_df, source_criteria,
                                   target_changes, preserve_attributes,
                                   website_ids, alpha_values=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0]):
    """Run batch synthesis experiment for all specified website IDs"""

    print(f"\n=== Batch Synthesis Experiment ===")
    print(f"Source criteria: {source_criteria}")
    print(f"Target changes: {target_changes}")
    print(f"Preserve attributes: {preserve_attributes}")
    print(f"Website IDs: {len(website_ids)} websites")

    # Select source samples for all specified websites
    source_sequences, source_metadata, source_df = select_source_samples(
        test_df, source_criteria, website_ids
    )

    if len(source_sequences) == 0:
        print("No samples found matching the criteria")
        return None, None

    source_sequences_tf = tf.constant(source_sequences, dtype=tf.float32)
    total_samples = len(source_sequences)

    # Test different alpha values
    best_alpha = None
    best_valid_count = 0

    for alpha in alpha_values:
        print(f"\n--- Testing alpha = {alpha} ---")

        # Run synthesis
        synthesized_sequences, valid_mask, scores = synthesizer.batch_synthesize(
            source_sequences_tf, target_changes, alpha=alpha,
            preserve_attributes=preserve_attributes, preserve_threshold=0.5
        )

        valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.int32)).numpy()
        print(
            f"Valid samples with alpha={alpha}: {valid_count}/{total_samples}")

        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_alpha = alpha

            # Keep best results
            best_synthesized = synthesized_sequences
            best_valid_mask = valid_mask
            best_scores = scores

    if best_alpha is None or best_valid_count == 0:
        print("No valid samples generated with any alpha value")
        return None, None

    print(f"\nBest alpha: {best_alpha} with {best_valid_count} valid samples")

    # Extract valid samples
    valid_indices = tf.where(best_valid_mask).numpy().flatten()
    valid_sequences = tf.gather(best_synthesized, valid_indices).numpy()

    # Create metadata for synthesized samples
    synth_metadata = create_synthesized_metadata(
        source_metadata, target_changes, valid_indices
    )

    # Evaluate synthesis quality
    quality_metrics = synthesizer.evaluate_synthesis_quality(
        tf.gather(source_sequences_tf, valid_indices),
        tf.gather(best_synthesized, valid_indices),
        target_changes
    )

    # Save synthesized dataset
    filepath = save_synthesized_dataset(
        valid_sequences, synth_metadata, target_changes, quality_metrics
    )

    # Print summary statistics
    print(f"\n=== Synthesis Summary ===")
    print(f"Original samples: {total_samples}")
    print(f"Valid synthesized samples: {best_valid_count}")
    print(f"Success rate: {best_valid_count/total_samples*100:.1f}%")
    print(f"Best alpha value: {best_alpha}")
    print(f"Websites covered: {len(synth_metadata['Website'].unique())}")

    for attr_name, scores_array in best_scores.items():
        valid_scores = scores_array[valid_indices]
        print(
            f"{attr_name} scores: mean={np.mean(valid_scores):.3f}, std={np.std(valid_scores):.3f}")

    return filepath, quality_metrics


if __name__ == "__main__":
    print("=== Batch Multi-Attribute Network Traffic Synthesis ===")

    # Load models and data
    vae_model, discriminators, attribute_names, test_website_ids, test_df = load_models_and_data()
    print(f"Length of test website IDs: {len(test_website_ids)}")

    # Create batch synthesizer
    synthesizer = BatchMultiAttributeSynthesizer(vae_model, discriminators)

    print(f"Available attributes: {attribute_names}")
    print(f"Available discriminators: {list(discriminators.keys())}")

    # Define synthesis experiments
    experiments = [
        {
            'name': 'leuven_to_singapore',
            'source_criteria': {
                'location': 'leuven',
                'client': 'cloudflare',
                'resolver': 'cloudflare',
                'platform': 'desktop'
            },
            'target_changes': {
                'location_singapore': 1.0,
            },
            'preserve_attributes': ['client_cloudflare', 'resolver_cloudflare']
        },
    ]

    # Run experiments
    results = []
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*60}")

        try:
            filepath, metrics = run_batch_synthesis_experiment(
                synthesizer, test_df,
                exp['source_criteria'], exp['target_changes'],
                exp['preserve_attributes'], test_website_ids
            )
            if filepath is not None:
                results.append({
                    'name': exp['name'],
                    'filepath': filepath,
                    'metrics': metrics
                })
        except Exception as e:
            print(f"Experiment {exp['name']} failed: {e}")
            continue

    print(f"\n{'='*60}")
    print("All Batch Synthesis Experiments Complete")
    print(f"{'='*60}")

    for result in results:
        print(f"✓ {result['name']}: {result['filepath']}")
