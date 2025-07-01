# downstream_classifier.py

from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder

from init_dataset import get_train_test_dataset

# --- Configuration Paths (Mirroring main synthesis script for consistency) ---
DATASET_CSV = '../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv'
BASE_OUTPUT_DIR = '../../dataset/synthesized/'  # Where synthetic data is saved

# Model parameters (should match synthesis configuration)
SEQUENCE_LENGTH = 32

# --- Classifier Model Definition ---


def build_1d_cnn_classifier(input_shape: Tuple[int,], num_classes: int):
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
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    # BinaryCrossentropy is suitable for binary classification with sigmoid output
    # Metrics include accuracy, precision, and recall for comprehensive evaluation
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()  # Print model summary to console
    return model

# --- Data Preparation for Classifier ---


def prepare_classifier_data(df: pd.DataFrame, sequence_length: int, label_encoder: Optional[LabelEncoder] = None):
    """
    Prepares sequences and website labels for the classifier from a DataFrame.
    Labels are encoded website IDs.
    """
    feature_cols = [str(i) for i in range(sequence_length)]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df['Website'].values

    # Fit or use provided label encoder
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)

    X = X.reshape(-1, sequence_length, 1)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    print(
        f"Prepared data: X shape {X.shape}, y shape {y.shape} (num classes: {len(np.unique(y))})")
    return X, y, label_encoder


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

    label_encoder = LabelEncoder()
    label_encoder.fit(test_df['Website'].values)
    num_classes = len(label_encoder.classes_)

    # --- Phase 1: Baseline Test (Train on Real 'Leuven', Test on Real 'Singapore') ---
    print("\n" + "="*60)
    print("Phase 1: Baseline Classification (Train on Real 'Leuven', Test on Real 'Singapore')")
    print("Objective: How well can a classifier trained on real 'Leuven' data identify real 'Singapore' data.")
    print("="*60)

    # Prepare training and testing data
    X_train_baseline, y_train_baseline, _ = prepare_classifier_data(
        df=test_df[test_df['Location'].str.lower() == 'leuven'],
        sequence_length=SEQUENCE_LENGTH,
        label_encoder=label_encoder
    )
    X_test_real, y_test_real, _ = prepare_classifier_data(
        df=test_df[test_df['Location'].str.lower() == 'singapore'],
        sequence_length=SEQUENCE_LENGTH,
        label_encoder=label_encoder
    )

    print("\nTraining baseline classifier on real 'Leuven' website data...")
    # Build and train the classifier
    baseline_classifier = build_1d_cnn_classifier(
        input_shape=(SEQUENCE_LENGTH, 1), num_classes=num_classes)
    baseline_classifier.fit(X_train_baseline, y_train_baseline,
                            epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    print("\nEvaluating baseline classifier on real 'Singapore' website data...")
    y_pred_baseline_probs = baseline_classifier.predict(X_test_real)
    y_pred_baseline = np.argmax(y_pred_baseline_probs, axis=1)

    print(f"\n--- Baseline Results (Train on Real 'Leuven', Test on Real 'Singapore' Websites) ---")
    # Generate classification report, including unseen classes if any in test set
    # inverse_transform is used to map integer labels back to original website names for readability
    print(classification_report(y_test_real, y_pred_baseline,
                                # Ensure all possible classes are represented
                                labels=np.arange(num_classes),
                                # Use the full set of original website names
                                target_names=label_encoder.classes_,
                                zero_division=0))
    print("=" * 60)

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

        # Filter synthetic data to only include websites that are known by our LabelEncoder
        # (i.e., those present in the original test_df)
        df_train_synthetic = df_synthetic[df_synthetic['Website'].isin(
            website_encoder.classes_)].copy()

        # Prepare training data: Synthetic data for Website classification
        X_train_synthetic, y_train_synthetic = prepare_classifier_data(
            df=df_train_synthetic,
            sequence_length=SEQUENCE_LENGTH,
            label_encoder=label_encoder
        )

        # Prepare testing data: Use the SAME real 'Singapore' data from Phase 1

        if X_train_synthetic.shape[0] == 0 or X_test_real.shape[0] == 0:
            print("Insufficient data for synthetic test. Skipping Phase 2.")
            print(
                f"Synthetic training samples available: {X_train_synthetic.shape[0]}")
            print(
                f"Real Singapore test samples available: {X_test_real.shape[0]}")
        else:
            synthetic_classifier = build_1d_cnn_classifier(
                input_shape=(SEQUENCE_LENGTH, 1), num_classes=num_classes
            )

            print("\nTraining classifier on synthetic 'Website' data...")
            synthetic_classifier.fit(X_train_synthetic, y_train_synthetic,
                                     epochs=10, batch_size=32, validation_split=0.2, verbose=1)

            print(
                "\nEvaluating classifier on real 'Singapore' website data (same test set as baseline)...")
            y_pred_synthetic_probs = synthetic_classifier.predict(X_test_real)
            y_pred_synthetic = np.argmax(y_pred_synthetic_probs, axis=1)

            print(f"\n--- Synthetic Data Results (Train on Synthetic 'Singapore' Websites, Test on Real 'Singapore' Websites) ---")
            print(classification_report(y_test_real, y_pred_synthetic,
                                        labels=np.arange(num_classes),
                                        target_names=label_encoder.classes_,
                                        zero_division=0))
            print("-------------------------------------------------------------------")

    print("\n=== Downstream Classifier Evaluation Complete ===")
