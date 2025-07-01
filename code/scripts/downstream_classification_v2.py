# downstream_classifier.py

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional

from init_dataset import get_train_test_dataset

# --- Configuration Paths (Mirroring main synthesis script for consistency) ---
DATASET_CSV = '../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv'
BASE_OUTPUT_DIR = '../../dataset/synthesized/'  # Where synthetic data is saved

# Model parameters (should match synthesis configuration)
SEQUENCE_LENGTH = 32

# --- Classifier Model Definition ---


def build_1d_cnn_classifier(input_shape: Tuple[int,], num_classes: int = 1):
    """
    Builds a simple 1D CNN classifier model for binary classification.
    Input shape should be (sequence_length, 1) for 1D convolution.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Conv1D layer to capture local patterns in the sequence
        tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                               activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),  # Reduce dimensionality

        # Another Conv1D layer for higher-level features
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               activation='relu', padding='same'),
        # Further reduce dimensionality
        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Flatten(),  # Flatten the output for dense layers

        # Dense layers for classification
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Dropout for regularization

        # Output layer: Sigmoid for binary classification (num_classes=1)
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile the model
    # BinaryCrossentropy is suitable for binary classification with sigmoid output
    # Metrics include accuracy, precision, and recall for comprehensive evaluation
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.summary()  # Print model summary to console
    return model

# --- Data Preparation for Classifier ---


def prepare_classifier_data(df: pd.DataFrame,
                            positive_class_location: str,
                            sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares sequences and binary labels for the classifier from a DataFrame.
    Samples from 'positive_class_location' are labeled 1, others are labeled 0.
    """
    feature_cols = [str(i) for i in range(sequence_length)]

    # Extract sequences (X) and convert to float32
    X = df[feature_cols].values.astype(np.float32)

    # Create binary labels (y): 1 if location matches positive_class_location, else 0
    # Ensure case-insensitive comparison for location names
    y = (df['Location'].str.lower() ==
         positive_class_location.lower()).astype(np.int32).values

    # Reshape X for Conv1D: (batch_size, sequence_length, features_per_step)
    # Each packet size is a single feature at each time step, so features_per_step = 1
    X = X.reshape(-1, sequence_length, 1)

    # Shuffle the data to ensure randomness in batches during training
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    print(
        f"Prepared data: X shape {X.shape}, y shape {y.shape} (Positive class: '{positive_class_location}')")
    return X, y


# --- Main Execution Block ---
if __name__ == "__main__":
    print("=== Downstream Classifier Evaluation ===")

    # Load the full original dataset to get the test_df based on unseen website IDs
    # We only need test_df, not train_ds or train_website_ids for this evaluation.
    # The get_train_test_dataset function is used here primarily to get the `test_df`
    # which is filtered by `test_website_ids` (unseen websites).
    print("Loading dataset to identify test split...")
    _, _, _, test_website_ids, _ = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1, length=SEQUENCE_LENGTH)

    df_original = pd.read_csv(DATASET_CSV, index_col=0)
    # Filter original dataset to include only traces from unseen websites (test split)
    test_df = df_original[df_original["Website"].isin(
        test_website_ids)].reset_index(drop=True)
    print(f"Total samples in test_df (from unseen websites): {len(test_df)}")

    # --- Phase 1: Baseline Test (Train on Real 'Leuven', Test on Real 'Singapore') ---
    print("\n" + "="*60)
    print("Phase 1: Baseline Classification (Train on Real 'Leuven', Test on Real 'Singapore')")
    print("Objective: How well can a classifier trained on real 'Leuven' data identify real 'Singapore' data.")
    print("="*60)

    # Prepare training data for baseline: Real 'Leuven' data from test_df
    # Labels: 1 for 'Singapore', 0 for 'Leuven' (so all y_train_baseline will be 0 here)
    X_train_baseline, y_train_baseline = prepare_classifier_data(
        df=test_df[test_df['Location'].str.lower() == 'leuven'],
        # We are training to classify 'singapore', so leuven is negative
        positive_class_location='singapore',
        sequence_length=SEQUENCE_LENGTH
    )

    # Prepare testing data for baseline: Real 'Singapore' data from test_df
    # Labels: 1 for 'Singapore' (so all y_test_baseline will be 1 here)
    X_test_baseline, y_test_baseline = prepare_classifier_data(
        df=test_df[test_df['Location'].str.lower() == 'singapore'],
        positive_class_location='singapore',  # Singapore is the positive class
        sequence_length=SEQUENCE_LENGTH
    )

    if X_train_baseline.shape[0] == 0 or X_test_baseline.shape[0] == 0:
        print("Insufficient real data for baseline test. Skipping Phase 1.")
    else:
        baseline_classifier = build_1d_cnn_classifier(
            input_shape=(SEQUENCE_LENGTH, 1), num_classes=1)

        print("\nTraining baseline classifier on real 'Leuven' data...")
        # Train the classifier. The model will learn to distinguish 'Singapore' (positive) from 'Leuven' (negative).
        # Since X_train_baseline only contains 'Leuven' data, all y_train_baseline are 0.
        # This is a challenging setup: learning "not Singapore" from one location to predict "is Singapore".
        baseline_classifier.fit(X_train_baseline, y_train_baseline,
                                epochs=10, batch_size=32, validation_split=0.2, verbose=1)

        print("\nEvaluating baseline classifier on real 'Singapore' data...")
        # Evaluate on the real Singapore data. Here, y_test_baseline are all 1s.
        loss, accuracy, precision, recall = baseline_classifier.evaluate(
            X_test_baseline, y_test_baseline, verbose=0)

        print(
            f"\n--- Baseline Results (Train on Real 'Leuven', Test on Real 'Singapore') ---")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("-------------------------------------------------------------------")

    # --- Phase 2: Synthetic Data Test (Train on Synthetic 'Singapore', Test on Real 'Singapore') ---
    print("\n" + "="*60)
    print("Phase 2: Synthetic Data Classification (Train on Synthetic 'Singapore', Test on Real 'Singapore')")
    print("Objective: How well can a classifier trained on synthetic 'Singapore' data identify real 'Singapore' data.")
    print("="*60)

    # Define the path to the synthetic data file
    synthetic_filename = "synthesized_location-singapore.csv"
    synthetic_filepath = os.path.join(BASE_OUTPUT_DIR, synthetic_filename)

    if not os.path.exists(synthetic_filepath):
        print(f"Synthetic data file not found: {synthetic_filepath}")
        print("Please run the main synthesis script (e.g., `python your_synthesis_script.py`) to generate it first.")
    else:
        # Load synthetic data
        print(f"Loading synthetic data from: {synthetic_filepath}")
        df_synthetic = pd.read_csv(synthetic_filepath)

        # Prepare training data for synthetic test: Synthetic 'Singapore' data
        # All synthetic samples are intended to be 'Singapore', so labels will be 1.
        X_train_synthetic, y_train_synthetic = prepare_classifier_data(
            df=df_synthetic,
            positive_class_location='singapore',  # All synthetic samples are 'singapore'
            sequence_length=SEQUENCE_LENGTH
        )

        # Prepare testing data for synthetic test: Use the SAME real 'Singapore' data as baseline
        # X_test_baseline and y_test_baseline are already prepared from Phase 1

        if X_train_synthetic.shape[0] == 0 or X_test_baseline.shape[0] == 0:
            print("Insufficient data for synthetic test. Skipping Phase 2.")
        else:
            synthetic_classifier = build_1d_cnn_classifier(
                input_shape=(SEQUENCE_LENGTH, 1), num_classes=1)

            print("\nTraining classifier on synthetic 'Singapore' data...")
            # Train the classifier using only the synthetic data
            synthetic_classifier.fit(X_train_synthetic, y_train_synthetic,
                                     epochs=10, batch_size=32, validation_split=0.2, verbose=1)

            print("\nEvaluating classifier on real 'Singapore' data...")
            # Evaluate on the real Singapore data
            loss_synth, accuracy_synth, precision_synth, recall_synth = synthetic_classifier.evaluate(
                X_test_baseline, y_test_baseline, verbose=0)

            print(
                f"\n--- Synthetic Data Results (Train on Synthetic 'Singapore', Test on Real 'Singapore') ---")
            print(f"Loss: {loss_synth:.4f}")
            print(f"Accuracy: {accuracy_synth:.4f}")
            print(f"Precision: {precision_synth:.4f}")
            print(f"Recall: {recall_synth:.4f}")
            print("-------------------------------------------------------------------")

    print("\n=== Downstream Classifier Evaluation Complete ===")
