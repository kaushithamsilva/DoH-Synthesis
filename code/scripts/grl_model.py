import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import triplet_functions  # Assuming this module provides the baseCNN


class GradientReversalLayer(layers.Layer):
    """
    Gradient Reversal Layer for domain adaptation.
    This layer reverses the gradient during backpropagation.
    """

    def __init__(self, grl_lambda=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_ = grl_lambda

    def call(self, inputs):
        # During the forward pass, we just return the inputs
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GradientReversalLayer, self).get_config()
        config.update({"lambda_": self.lambda_})
        return config

    def compute_gradients(self, loss, variables):
        grads = super().compute_gradients(loss, variables)
        # Reverse the gradients
        for grad in grads:
            if grad is not None:
                grad *= -self.lambda_
        return grads


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

    # 4) Label classifier head: Predicts the class labels.
    label_preds = layers.Dense(
        num_classes, activation='softmax', name='label_classifier'
    )(features)  # Use normalized_features

    # 5) Domain classifier via GRL: Predicts the source domain/location.
    # encouraging the feature extractor to learn domain-invariant features.
    x_grl = GradientReversalLayer(grl_lambda=grl_lambda)(
        features)  # GRL on normalized_features

    # A small dense layer before the final domain classification output.
    x = layers.Dense(64, activation='relu')(x_grl)

    # Final domain classification layer.
    domain_preds = layers.Dense(
        num_locations, activation='softmax', name='domain_classifier')(x)

    # Define the Keras Model with shared input and two distinct outputs.
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
    assert set(np.unique(d)).issubset(set(range(num_locations))
                                      ), f"Domain labels 'd' should be integers from 0 to {num_locations-1}"

    grl_lambda = 0.01  # The scaling factor for the reversed gradient

    # Build the GRL model
    model = build_grl_model(input_dim, num_classes, num_locations, grl_lambda)

    # Configure the Adam optimizer with a learning rate and gradient clipping
    # Reduced learning rate and added clipvalue for better stability
    opt = tf.keras.optimizers.Adam(
        learning_rate=5e-5, clipvalue=0.5)

    # Compile the model with respective loss functions, loss weights, and metrics
    model.compile(
        optimizer=opt,
        loss={
            'label_classifier': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
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
        batch_size=64,
        epochs=200,
        shuffle=True,  # Shuffle data before each epoch
        callbacks=[tf.keras.callbacks.TerminateOnNaN()]
    )

    # Save the trained model in Keras format
    model.save(
        f"../../models-{locations[0]}-{locations[1]}/website/grl_model.keras")
