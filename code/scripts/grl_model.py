import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import triplet_functions  # Assuming this module provides the baseCNN


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, grl_lambda=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.grl_lambda = grl_lambda

    def call(self, x):
        @tf.custom_gradient
        def grad_reverse(x):
            def custom_grad(dy):
                # Add gradient clipping to prevent extreme values
                clipped_dy = tf.clip_by_value(dy, -1.0, 1.0)
                return -self.grl_lambda * clipped_dy
            return x, custom_grad
        return grad_reverse(x)


def build_grl_model(input_dim, num_classes, num_locations, grl_lambda=1.0):
    """
    Builds a Keras model with a Gradient Reversal Layer for domain adaptation.
    """
    # Input
    inputs = keras.Input(shape=(input_dim,), name="feature_input")

    # Expand to 3D for Conv1D
    x = layers.Reshape((input_dim, 1))(inputs)

    # Feature extraction
    features = triplet_functions.baseCNN(input_dim)(x)

    # Add batch normalization and dropout for stability
    features_norm = layers.BatchNormalization()(features)
    features_dropout = layers.Dropout(0.3)(features_norm)

    # Label classifier head
    label_preds = layers.Dense(
        num_classes, activation='softmax', name='label_classifier'
    )(features_dropout)

    # Domain classifier via GRL with improved architecture
    x_grl = GradientReversalLayer(grl_lambda=grl_lambda)(features_norm)

    # More robust domain classifier architecture
    x = layers.Dense(128, activation='relu')(x_grl)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Final domain classification with explicit numerical stability
    domain_logits = layers.Dense(num_locations, name='domain_logits')(x)
    domain_preds = layers.Softmax(name='domain_classifier')(domain_logits)

    return keras.Model(
        inputs=inputs,
        outputs=[label_preds, domain_preds],
        name='feature_extractor_grl'
    )


if __name__ == '__main__':
    # Import necessary modules for running the example
    import init_gpu
    import init_dataset
    import pandas as pd
    import numpy as np

    # Initialize GPUs if available
    init_gpu.initialize_gpus()

    # Define locations for dataset loading
    locations = ['LOC2', 'LOC3']

    # Load the processed and scaled dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    # Get sample data for training and testing
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # Determine the input dimension based on the training data (excluding 'Website' and 'Location' columns)
    input_dim = train_df.shape[1] - 2
    print(f"Input dimension: {input_dim}")

    train_df = pd.concat(
        [train_df, test_df[test_df['Location'] == locations[0]]])

    # Prepare input features (X), website labels (y), and domain labels (d)
    X = train_df.iloc[:, 2:].to_numpy().astype(np.float32)
    y = train_df['Website'].to_numpy().astype(int)
    # Convert 'Location' strings to integer labels (0 for source, 1 for target)
    d = train_df['Location'].apply(
        lambda x: 0 if x == locations[0] else 1).to_numpy().astype(int)

    # Assertions to ensure data integrity
    assert set(np.unique(d)).issubset(
        {0, 1}), "Domain labels should only be 0 or 1."
    assert not np.any(np.isnan(d)), "Domain labels contain NaN values."

    # Explicit check for NaNs/Infs in the input features X
    if not np.all(np.isfinite(X)):  # More robust check for NaNs and Infs
        print("Warning: Input data X contains NaN or Inf values! This can lead to unstable training.")
        # Consider adding data cleaning steps here if this warning appears frequently.

    if not np.all(np.isfinite(X)):
        print("ERROR: Input data X contains NaN or Inf values!")
        num_nans = np.sum(np.isnan(X))
        num_infs = np.sum(np.isinf(X))
        print(
            f"Number of NaNs in X: {num_nans}, Number of Infs in X: {num_infs}")
        # Consider handling these, e.g., by imputation or removing problematic rows/columns
    else:
        print("Input data X is finite (no NaNs or Infs).")

    # Dynamically determine num_classes and num_locations from the data
    num_classes = len(np.unique(y))
    num_locations = len(np.unique(d))

    print(
        f"Unique y labels: {np.unique(y)}, Max y: {np.max(y)}, Num classes: {num_classes}")
    assert np.all(y >= 0) and np.all(
        y < num_classes), "Class labels 'y' are out of range!"

    print(
        f"Unique d labels: {np.unique(d)}, Max d: {np.max(d)}, Num locations: {num_locations}")
    assert np.all(d >= 0) and np.all(
        d < num_locations), "Domain labels 'd' are out of range!"
    assert set(np.unique(d)).issubset(set(range(num_locations)))

    # FIXED PARAMETERS - Key changes for stability
    grl_lambda = 0.1  # Increased from 0.01 for better gradient flow

    # Build the GRL model
    model = build_grl_model(input_dim, num_classes, num_locations, grl_lambda)

    # IMPROVED OPTIMIZER CONFIGURATION
    opt = tf.keras.optimizers.Adam(
        learning_rate=1e-4,  # Reduced learning rate
        clipnorm=1.0,        # Gradient norm clipping instead of value clipping
        epsilon=1e-7         # Smaller epsilon for numerical stability
    )

    # IMPROVED LOSS CONFIGURATION
    model.compile(
        optimizer=opt,
        loss={
            'label_classifier': tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, label_smoothing=0.01  # Add label smoothing
            ),
            'domain_classifier': tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, label_smoothing=0.01  # Add label smoothing
            ),
        },
        loss_weights={
            'label_classifier': 1.0,
            'domain_classifier': grl_lambda  # Match GRL lambda for consistency
        },
        metrics={
            'label_classifier': ['accuracy', 'top_5_accuracy'],
            'domain_classifier': ['accuracy']
        }
    )

    print(model.summary())

    # IMPROVED TRAINING CONFIGURATION
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5, min_lr=1e-7
        )
    ]

    # Train with smaller batch size for stability
    history = model.fit(
        X,
        {'label_classifier': y, 'domain_classifier': d},
        batch_size=32,  # Reduced batch size
        epochs=200,
        validation_split=0.1,  # Add validation monitoring
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Save the trained model
    model.save(
        f"../../models-{locations[0]}-{locations[1]}/website/grl_model.keras")
