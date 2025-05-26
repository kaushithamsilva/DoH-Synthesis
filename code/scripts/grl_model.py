import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import triplet_functions  # Assuming this module provides the baseCNN


class GradientReversalLayer(keras.layers.Layer):
    """
    Implements the Gradient Reversal Layer (GRL).
    During the forward pass, it acts as an identity function.
    During the backward pass, it reverses the sign of the gradients
    and scales them by `lambda_`.
    """

    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def call(self, x):
        """
        Applies the custom gradient function to the input `x`.
        """
        @tf.custom_gradient
        def reverse_gradient(x):
            """
            Defines the forward and backward pass for the GRL.
            """
            def grad(dy):
                """
                The gradient function for the backward pass.
                It reverses the sign of the incoming gradient `dy`
                and scales it by `self.lambda_`.
                """
                return -self.lambda_ * dy
            return x, grad  # Forward pass returns x, backward pass uses grad
        return reverse_gradient(x)


def build_grl_model(input_dim, num_classes, num_locations, grl_lambda=1.0):
    """
    Builds a Keras model with a Gradient Reversal Layer for domain adaptation.

    Args:
        input_dim (int): The dimensionality of the input features.
        num_classes (int): The number of classes for the main classification task.
        num_locations (int): The number of domains/locations for the domain classification task.
        grl_lambda (float): The lambda parameter for the Gradient Reversal Layer,
                            controlling the strength of gradient reversal.

    Returns:
        keras.Model: A Keras Model with two outputs: label predictions and domain logits.
    """
    # 1) Input is 2D (batch_size, input_dim)
    inputs = keras.Input(shape=(input_dim,), name="feature_input")

    # 2) Expand to 3D for Conv1D: (batch_size, input_dim, 1)
    # baseCNN is assumed to be a Conv1D-based network, which typically expects 3D input.
    x = layers.Reshape((input_dim, 1))(inputs)

    # 3) Pass through your baseCNN to extract features.
    # The output of baseCNN is expected to be 2D (batch_size, feature_dim).
    features = triplet_functions.baseCNN(input_dim)(x)

    # Add BatchNormalization after baseCNN to stabilize the features.
    # This is crucial for preventing NaN values by normalizing activations
    # and making the training more robust.
    normalized_features = layers.BatchNormalization(
        name='normalized_features')(features)

    # 4) Label classifier head: Predicts the class labels.
    # It takes the normalized features and outputs probabilities using softmax.
    label_preds = layers.Dense(
        num_classes, activation='softmax', name='label_classifier'
    )(normalized_features)  # Use normalized_features

    # 5) Domain classifier via GRL: Predicts the source domain/location.
    # The GRL is applied to the normalized features.
    # It reverses the gradient flow for the domain classifier,
    # encouraging the feature extractor to learn domain-invariant features.
    x_grl = GradientReversalLayer(lambda_=grl_lambda)(
        normalized_features)  # GRL on normalized_features

    # A small dense layer before the final domain classification output.
    x = layers.Dense(64, activation='relu')(x_grl)

    # Final domain classification layer.
    # Changed to use 'softmax' activation as requested.
    domain_logits = layers.Dense(
        num_locations, activation='softmax', name='domain_classifier')(x)

    # Define the Keras Model with shared input and two distinct outputs.
    return keras.Model(
        inputs=inputs,
        outputs=[label_preds, domain_logits],
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

    # IMPORTANT!: Append the source data from the test set to the training set.
    # This is a common practice in domain adaptation where the source domain
    # data is often augmented or used more extensively.
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

    # Dynamically determine num_classes and num_locations from the data
    num_classes = len(np.unique(y))
    num_locations = len(np.unique(d))

    grl_lambda = 0.1  # The scaling factor for the reversed gradient

    # Build the GRL model
    model = build_grl_model(input_dim, num_classes, num_locations, grl_lambda)

    # Configure the Adam optimizer with a learning rate and gradient clipping
    # Reduced learning rate and added clipvalue for better stability
    opt = tf.keras.optimizers.Adam(
        learning_rate=5e-5, clipnorm=1.0, clipvalue=0.5)  # Adjusted learning_rate

    # Compile the model with respective loss functions, loss weights, and metrics
    model.compile(
        optimizer=opt,
        loss={
            # For label classification, use SparseCategoricalCrossentropy.
            # `from_logits=False` because the `label_classifier` dense layer uses `softmax` activation.
            'label_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            # For domain classification, use SparseCategoricalCrossentropy.
            # Changed `from_logits=True` to `from_logits=False` because the `domain_classifier`
            # now uses `softmax` activation.
            'domain_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        },
        loss_weights={
            # Weight for the label classification loss
            'label_classifier': 1.0,
            # Weight for the domain classification loss (often lower to balance objectives)
            'domain_classifier': 0.1
        },
        metrics={
            # Metrics to monitor during training for both tasks
            'label_classifier': 'accuracy',
            'domain_classifier': 'accuracy'
        }
    )

    # Print a summary of the model architecture, including the new BatchNormalization layer
    print(model.summary())

    # Train the model
    # X is the input features, and the outputs are provided as a dictionary
    # mapping output layer names to their respective ground truth labels.
    model.fit(
        X,
        {'label_classifier': y, 'domain_classifier': d},
        batch_size=128,
        epochs=200,
        shuffle=True  # Shuffle data before each epoch
    )

    # Save the trained model in Keras format
    model.save(
        f"../../models-{locations[0]}-{locations[1]}/website/grl_model.keras")
