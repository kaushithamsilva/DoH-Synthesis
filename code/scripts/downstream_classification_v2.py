# downstream_classifier.py

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
# Correct import for sklearn's report
from sklearn.metrics import classification_report

# Make sure init_dataset is accessible in your project structure
from init_dataset import get_train_test_dataset

# --- Configuration Paths (Mirroring main synthesis script for consistency) ---
DATASET_CSV = '../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv'
BASE_OUTPUT_DIR = '../../dataset/synthesized/'  # Where synthetic data is saved

# Model parameters (should match synthesis configuration)
SEQUENCE_LENGTH = 32

# --- Classifier Model Definition ---


def build_1d_cnn_classifier(input_shape: Tuple[int,], num_classes: int):
    """
    Builds a simple 1D CNN classifier model for multi-class classification.
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

        # Output layer: Softmax for multi-class classification
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    # SparseCategoricalCrossentropy is suitable when labels are integers (not one-hot encoded)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()  # Print model summary to console
    return model

# --- Data Preparation for Classifier ---


def prepare_classifier_data(df: pd.DataFrame,
                            sequence_length: int,
                            label_encoder: LabelEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares sequences and integer-encoded 'Website' labels for the classifier from a DataFrame.
    It expects a pre-fitted LabelEncoder.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Returning empty arrays.")
        return np.array([]).reshape(0, sequence_length, 1), np.array([])

    feature_cols = [str(i) for i in range(sequence_length)]

    X = df[feature_cols].values.astype(np.float32)

    # Use the provided (already fitted) label_encoder to transform 'Website' names
    y_raw = df['Website'].values
    # This will raise an error if an unseen label is present
    y = label_encoder.transform(y_raw)

    X = X.reshape(-1, sequence_length, 1)

    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    print(
        f"Prepared data: X shape {X.shape}, y shape {y.shape} (Websites: {len(np.unique(y))} unique)")
    return X, y


# --- Main Execution Block ---
if __name__ == "__main__":
    print("=== Downstream Classifier Evaluation for 'Website' ===")

    # Load the full original dataset to get the test_df based on unseen website IDs
    print("Loading dataset to identify test split...")
    # This call gives us the test_website_ids (websites unseen during VAE/discriminator training)
    _, _, _, test_website_ids, _ = get_train_test_dataset(
        DATASET_CSV, num_train=1200, num_test=300, batch_size=1, length=SEQUENCE_LENGTH)

    df_original = pd.read_csv(DATASET_CSV, index_col=0)
    # Filter original dataset to include only traces from unseen websites (test split)
    test_df = df_original[df_original["Website"].isin(
        test_website_ids)].reset_index(drop=True)
    print(f"Total samples in test_df (from unseen websites): {len(test_df)}")

    # Fit LabelEncoder on ALL unique website IDs in the test_df
    # This ensures consistent encoding across all phases (baseline, synthetic train, real test)
    unique_websites_in_test_df = test_df['Website'].unique()
    if len(unique_websites_in_test_df) < 2:
        print("Error: Less than 2 unique websites in test_df. Cannot perform multi-class classification.")
        exit()  # Exit if classification is not possible

    website_encoder = LabelEncoder()
    website_encoder.fit(unique_websites_in_test_df)
    num_website_classes = len(website_encoder.classes_)
    print(
        f"Number of website classes (fitted on all test_df websites): {num_website_classes}")

    # --- Phase 1: Baseline Test (Train on Real 'Leuven', Test on Real 'Singapore' for Website Classification) ---
    print("\n" + "="*60)
    print("Phase 1: Baseline Classification (Train on Real 'Leuven' Website Data, Test on Real 'Singapore' Website Data)")
    print("Objective: Evaluate cross-location generalization for Website classification on real data.")
    print("="*60)

    # Prepare training data: Real 'Leuven' data from test_df
    df_train_baseline_leuven = test_df[test_df['Location'].str.lower(
    ) == 'leuven'].copy()
    X_train_baseline, y_train_baseline = prepare_classifier_data(
        df=df_train_baseline_leuven,
        sequence_length=SEQUENCE_LENGTH,
        label_encoder=website_encoder
    )

    # Prepare testing data: Real 'Singapore' data from test_df
    df_test_real_singapore = test_df[test_df['Location'].str.lower(
    ) == 'singapore'].copy()
    X_test_real_singapore, y_test_real_singapore = prepare_classifier_data(
        df=df_test_real_singapore,
        sequence_length=SEQUENCE_LENGTH,
        label_encoder=website_encoder
    )

    if X_train_baseline.shape[0] == 0 or X_test_real_singapore.shape[0] == 0:
        print("Insufficient real data for baseline test after location filtering. Skipping Phase 1.")
        print(
            f"Samples in train (Leuven): {X_train_baseline.shape[0]}, Samples in test (Singapore): {X_test_real_singapore.shape[0]}")
    else:
        # Check for common websites in train and test sets for meaningful classification
        train_websites_in_baseline = df_train_baseline_leuven['Website'].unique(
        )
        test_websites_in_baseline = df_test_real_singapore['Website'].unique()
        common_websites_baseline = np.intersect1d(
            train_websites_in_baseline, test_websites_in_baseline)
        if len(common_websites_baseline) == 0:
            print("\nWARNING: No common websites found between 'Leuven' training data and 'Singapore' testing data for baseline.")
            print("Website classification results in this setup might be very low as the model won't have seen test classes.")
            print(f"Websites in Leuven train: {train_websites_in_baseline}")
            print(f"Websites in Singapore test: {test_websites_in_baseline}")

        baseline_classifier = build_1d_cnn_classifier(input_shape=(
            SEQUENCE_LENGTH, 1), num_classes=num_website_classes)

        print("\nTraining baseline classifier on real 'Leuven' website data...")
        baseline_classifier.fit(X_train_baseline, y_train_baseline,
                                epochs=10, batch_size=32, validation_split=0.2, verbose=1)

        print("\nEvaluating baseline classifier on real 'Singapore' website data...")
        y_pred_baseline_probs = baseline_classifier.predict(
            X_test_real_singapore)
        y_pred_baseline = np.argmax(y_pred_baseline_probs, axis=1)

        print(f"\n--- Baseline Results (Train on Real 'Leuven', Test on Real 'Singapore' Websites) ---")
        # **CORRECTION**: Convert label_encoder.classes_ to a list of strings explicitly
        print(classification_report(y_test_real_singapore, y_pred_baseline,
                                    # Ensure all possible classes are represented
                                    labels=np.arange(num_website_classes),
                                    # Convert to string
                                    target_names=[
                                        str(c) for c in website_encoder.classes_],
                                    zero_division=0))
        print("-------------------------------------------------------------------")

    # --- Phase 2: Synthetic Data Test (Train on Synthetic 'Singapore', Test on Real 'Singapore' for Website Classification) ---
    print("\n" + "="*60)
    print("Phase 2: Synthetic Data Classification (Train on Synthetic 'Singapore' Website Data, Test on Real 'Singapore' Website Data)")
    print("Objective: How well can a classifier trained on synthetic 'Singapore' data identify real website traces.")
    print("="*60)

    # Define the path to the synthetic data file
    # This file should contain the original 'Website' column to be used as labels
    synthetic_filename = "synthesized_location-singapore.csv"
    synthetic_filepath = os.path.join(BASE_OUTPUT_DIR, synthetic_filename)

    if not os.path.exists(synthetic_filepath):
        print(f"Synthetic data file not found: {synthetic_filepath}")
        print("Please ensure your synthesis script (e.g., `run_batch_synthesis_experiment`) has generated it.")
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
            label_encoder=website_encoder
        )

        # Prepare testing data: Use the SAME real 'Singapore' data from Phase 1
        # X_test_real_singapore and y_test_real_singapore are already prepared

        if X_train_synthetic.shape[0] == 0 or X_test_real_singapore.shape[0] == 0:
            print("Insufficient data for synthetic test. Skipping Phase 2.")
            print(
                f"Synthetic training samples available: {X_train_synthetic.shape[0]}")
            print(
                f"Real Singapore test samples available: {X_test_real_singapore.shape[0]}")
        else:
            # Check for common websites between synthetic train and real Singapore test sets
            synthetic_train_websites = df_train_synthetic['Website'].unique()
            common_websites_synthetic_test = np.intersect1d(
                synthetic_train_websites, df_test_real_singapore['Website'].unique())
            if len(common_websites_synthetic_test) == 0:
                print(
                    "\nWARNING: No common websites found between synthetic training data and real 'Singapore' testing data.")
                print("Website classification results in this setup might be very low.")
                print(
                    f"Websites in synthetic train: {synthetic_train_websites}")
                print(
                    f"Websites in real Singapore test: {df_test_real_singapore['Website'].unique()}")

            synthetic_classifier = build_1d_cnn_classifier(
                input_shape=(SEQUENCE_LENGTH, 1), num_classes=num_website_classes
            )

            print("\nTraining classifier on synthetic 'Website' data...")
            synthetic_classifier.fit(X_train_synthetic, y_train_synthetic,
                                     epochs=10, batch_size=32, validation_split=0.2, verbose=1)

            print(
                "\nEvaluating classifier on real 'Singapore' website data (same test set as baseline)...")
            y_pred_synthetic_probs = synthetic_classifier.predict(
                X_test_real_singapore)
            y_pred_synthetic = np.argmax(y_pred_synthetic_probs, axis=1)

            print(f"\n--- Synthetic Data Results (Train on Synthetic 'Singapore' Websites, Test on Real 'Singapore' Websites) ---")
            # **CORRECTION**: Convert label_encoder.classes_ to a list of strings explicitly
            print(classification_report(y_test_real_singapore, y_pred_synthetic,
                                        labels=np.arange(num_website_classes),
                                        # Convert to string
                                        target_names=[
                                            str(c) for c in website_encoder.classes_],
                                        zero_division=0))
            print("-------------------------------------------------------------------")

    print("\n=== Downstream Classifier Evaluation Complete ===")
