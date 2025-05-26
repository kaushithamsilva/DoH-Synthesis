import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from grl_model import GradientReversal
import init_gpu
import init_dataset


if __name__ == "__main__":
    # --- Configuration matching train.py ---
    # init_gpu.initialize_gpus() # Uncomment if you have this module

    locations = ['LOC2', 'LOC3']

    # Define the path to your saved model
    model_save_path = f"../../models-{locations[0]}-{locations[1]}/website/dann_1d_sequence_model.keras"

    # --- Load Data (similar to train.py to get test_df) ---
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    test_df_target = test_df[test_df['Location'] == locations[1]]

    if test_df_target.empty:
        print(
            f"No target domain data found in test_df for location '{locations[1]}'.")
        print("Please ensure your data splitting logic correctly populates test_df with target samples.")
        exit()

    # Determine feature columns, input_dim, and num_classes as done in train.py
    feature_columns = [
        col for col in df.columns if col not in ['Website', 'Location']]
    input_dim = len(feature_columns)
    num_classes = df['Website'].nunique()

    # Model input shape (sequence_length, num_features)
    # This must match what the model was trained with
    sequence_length = input_dim
    num_features = 1
    input_shape = (sequence_length, num_features)

    # Prepare Target Test Data
    X_target_test = test_df_target[feature_columns].values.astype(np.float32)
    y_target_test = test_df_target['Website'].values.astype(np.int32)

    # Reshape X_target_test for Conv1D input
    X_target_test = X_target_test.reshape(-1, sequence_length, num_features)

    print(f"Loaded Target Test data shape: {X_target_test.shape}")
    print(f"Loaded Target Test labels shape: {y_target_test.shape}")
    print(f"Number of classes: {num_classes}")

    # --- Load the trained DANN model ---
    try:
        # Pass the custom GradientReversal layer to load_model
        custom_objects = {'GradientReversal': GradientReversal}
        dann_model = tf.keras.models.load_model(
            model_save_path, custom_objects=custom_objects)
        print(f"\nModel loaded successfully from: {model_save_path}")
        dann_model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Ensure the model path is correct: {model_save_path}")
        print("Also, ensure the GradientReversal class is defined or correctly imported.")
        exit()

    # --- Evaluate the model on Target Test Data ---
    print("\nEvaluating model on target domain test data...")
    # The model outputs two predictions: label_output and domain_output
    label_predictions_logits, domain_predictions_raw = dann_model.predict(
        X_target_test)

    # Get class predictions for labels (softmax output)
    y_pred_target = np.argmax(label_predictions_logits, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_target_test, y_pred_target)
    report = classification_report(
        y_target_test, y_pred_target, zero_division=0)

    print(f"\n--- Evaluation Results for Target Domain ({locations[1]}) ---")
    print(f"Accuracy on Target Test Data: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Optionally, you can also check the domain prediction on target data
    # The domain classifier is trying to predict 1 for target data.
    # If adaptation is successful, its accuracy should be around 0.5 (confused).
    domain_preds_target_binary = (domain_predictions_raw > 0.5).astype(int)
    target_domain_true_labels = np.ones_like(
        y_target_test).reshape(-1, 1)  # All are target domain
    domain_acc_target = accuracy_score(
        target_domain_true_labels, domain_preds_target_binary)
    print(
        f"\nDomain Classifier Accuracy on Target Test Data (should be ~0.5 for successful adaptation): {domain_acc_target:.4f}")
