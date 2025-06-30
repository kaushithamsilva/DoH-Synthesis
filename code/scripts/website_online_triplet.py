import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import logging
from typing import Tuple, Optional
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletSemiHardLossVectorized(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, name="triplet_semihard_loss_vectorized"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        labels = tf.cast(y_true, dtype=tf.int32)
        embeddings = y_pred

        if len(labels.shape) > 1 and labels.shape[1] == 1:
            labels = tf.squeeze(labels, axis=1)

        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)

        # Create masks
        batch_size = tf.shape(labels)[0]
        labels_equal = tf.equal(tf.expand_dims(
            labels, 0), tf.expand_dims(labels, 1))
        identity_mask = tf.eye(batch_size, dtype=tf.bool)

        positive_mask = tf.logical_and(
            labels_equal, tf.logical_not(identity_mask))
        negative_mask = tf.logical_not(labels_equal)

        # Get hardest positive for each anchor
        positive_distances = tf.where(
            positive_mask, distances, -tf.float32.max)
        hardest_positive = tf.reduce_max(positive_distances, axis=1)

        # Mask for semi-hard negatives
        semi_hard_mask = tf.logical_and(
            negative_mask,
            tf.logical_and(
                distances > tf.expand_dims(hardest_positive, 1),
                distances < tf.expand_dims(hardest_positive + self.margin, 1)
            )
        )

        # Get semi-hard negatives
        semi_hard_distances = tf.where(
            semi_hard_mask, distances, tf.float32.max)
        closest_semi_hard = tf.reduce_min(semi_hard_distances, axis=1)

        # If no semi-hard negatives, use hardest negative
        negative_distances = tf.where(negative_mask, distances, tf.float32.max)
        hardest_negative = tf.reduce_min(negative_distances, axis=1)

        chosen_negative = tf.where(
            tf.equal(closest_semi_hard, tf.float32.max),
            hardest_negative,
            closest_semi_hard
        )

        # Compute loss
        loss = tf.maximum(0.0, hardest_positive -
                          chosen_negative + self.margin)

        # Only consider valid triplets
        valid_mask = tf.logical_and(
            tf.not_equal(hardest_positive, -tf.float32.max),
            tf.not_equal(chosen_negative, tf.float32.max)
        )

        valid_losses = tf.boolean_mask(loss, valid_mask)

        if tf.size(valid_losses) == 0:
            return 0.0

        return tf.reduce_mean(valid_losses)

    def _pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distances between embeddings."""
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        square_norm = tf.linalg.diag_part(dot_product)

        distances_squared = (
            tf.expand_dims(square_norm, 1) -
            2.0 * dot_product +
            tf.expand_dims(square_norm, 0)
        )

        distances_squared = tf.maximum(distances_squared, 0.0)
        distances = tf.sqrt(distances_squared + tf.keras.backend.epsilon())

        return distances


class TripletTrainingConfig:
    """Configuration class for triplet training parameters."""

    def __init__(
        self,
        feature_length: int = 32,
        num_train_samples: int = 1200,
        batch_size: int = 128,
        epochs: int = 1000,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        margin: float = 0.2,
        patience: int = 50,
        validation_split: float = 0.2,
        base_network_name: str = 'baseCNN'
    ):
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


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int,
    feature_columns: Optional[list] = None,
    label_column: str = 'Website',
    shuffle_buffer: Optional[int] = None
) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from a DataFrame with better memory efficiency.

    Args:
        df: Input DataFrame
        batch_size: Batch size for training
        feature_columns: List of feature column names (if None, uses all except label)
        label_column: Name of the label column
        shuffle_buffer: Buffer size for shuffling (if None, uses dataset length)

    Returns:
        tf.data.Dataset ready for training
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != label_column]

    # Validate inputs
    if label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in DataFrame")

    if not all(col in df.columns for col in feature_columns):
        missing_cols = [
            col for col in feature_columns if col not in df.columns]
        raise ValueError(f"Feature columns not found: {missing_cols}")

    # Extract features and labels
    features = df[feature_columns].values.astype(np.float32)
    labels = df[label_column].values

    # Encode string labels to integers if needed
    if labels.dtype.kind in 'US':  # Unicode or byte string
        unique_labels = np.unique(labels)
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_int[label] for label in labels])
        logger.info(
            f"Encoded {len(unique_labels)} unique labels: {list(unique_labels)}")

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle with appropriate buffer size
    if shuffle_buffer is None:
        shuffle_buffer = len(df)
    dataset = dataset.shuffle(buffer_size=min(shuffle_buffer, len(df)))

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    logger.info(
        f"Created dataset with {len(df)} samples, batch size {batch_size}")
    return dataset


def split_dataset(
    dataset: tf.data.Dataset,
    validation_split: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split dataset into training and validation sets."""
    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    if total_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
        raise ValueError("Cannot determine dataset size for splitting")

    val_batches = int(total_batches * validation_split)
    train_batches = total_batches - val_batches

    train_dataset = dataset.take(train_batches)
    val_dataset = dataset.skip(train_batches)

    logger.info(
        f"Split dataset: {train_batches} training batches, {val_batches} validation batches")
    return train_dataset, val_dataset


def create_model_and_compile(
    base_network: tf.keras.Model,
    config: TripletTrainingConfig
) -> tf.keras.Model:
    """Create and compile the model with triplet loss."""

    # Create triplet loss
    triplet_loss = TripletSemiHardLossVectorized(margin=config.margin)

    # Create optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Compile model
    base_network.compile(
        optimizer=optimizer,
        loss=triplet_loss,
        # Add metrics for monitoring
        metrics=[triplet_loss]  # Track loss as a metric too
    )

    logger.info(
        f"Model compiled with margin={config.margin}, lr={config.learning_rate}")
    return base_network


def setup_callbacks(config: TripletTrainingConfig, model_save_path: str) -> list:
    """Setup training callbacks."""
    callbacks_list = []

    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks_list.append(early_stop)

    # Model checkpoint
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path.replace('.keras', '_best.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks_list.append(checkpoint)

    # Learning rate reduction
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=config.patience // 2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks_list.append(lr_scheduler)

    # CSV logger
    csv_logger = callbacks.CSVLogger(
        model_save_path.replace('.keras', '_training_log.csv')
    )
    callbacks_list.append(csv_logger)

    return callbacks_list


def main():
    """Main training function."""
    # Initialize configuration
    config = TripletTrainingConfig(
        feature_length=32,
        num_train_samples=1200,
        batch_size=128,
        epochs=1000,
        learning_rate=1e-4,
        weight_decay=1e-5,
        margin=0.2,
        patience=50,
        validation_split=0.2,
        base_network_name='baseCNN'
    )

    # Initialize GPU
    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2', 'LOC3']

    logger.info("Loading Dataset...")
    # Load the dataset
    dataset_path = f"../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, index_col=0)

    # Prepare data
    df.drop(columns=['Location', 'Resolver', 'Client',
            'Platform'], inplace=True, errors='ignore')
    feature_columns = [str(i) for i in range(config.feature_length)]
    df = df.loc[:, ['Website'] + feature_columns]

    # Get train-test split
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), config.num_train_samples
    )

    logger.info("Creating TensorFlow Datasets...")
    # Create full training dataset
    full_train_dataset = create_tf_dataset(
        train_df,
        batch_size=config.batch_size,
        feature_columns=feature_columns
    )

    # Split into train and validation
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        config.validation_split
    )

    # Clean up memory
    del df, train_df, test_df

    logger.info("Setting up model...")
    # Initialize base network
    base_instance = getattr(triplet_functions, config.base_network_name)(
        config.feature_length)

    # Create and compile model
    model = create_model_and_compile(base_instance, config)

    # Setup model save path
    model_save_path = (
        f'../../models/website/{"-".join(locations)}-{config.base_network_name}-'
        f'online_semi_hard_AdamW-epochs{config.epochs}-'
        f'train_samples{config.num_train_samples}-batch{config.batch_size}.keras'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Setup callbacks
    callbacks_list = setup_callbacks(config, model_save_path)

    logger.info("Starting training...")
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=callbacks_list,
        verbose=1
    )

    logger.info("Saving final model...")
    # Save the final model
    model.save(model_save_path)

    logger.info("Training completed successfully!")

    # Print training summary
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    logger.info(f"Final training loss: {final_loss:.4f}")
    logger.info(f"Final validation loss: {final_val_loss:.4f}")

    return model, history


if __name__ == '__main__':
    model, history = main()
