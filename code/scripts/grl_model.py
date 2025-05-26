import tensorflow as tf
import triplet_functions
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd

# Assuming these are available in your environment
# import init_gpu
# import init_dataset

# 1. Custom Gradient Reversal Layer (GRL)


class GradientReversal(layers.Layer):
    """
    A custom Keras layer that implements the Gradient Reversal Layer (GRL).
    During the forward pass, it acts as an identity function.
    During the backward pass, it multiplies the gradients by a negative scalar (alpha).
    """

    def __init__(self, alpha=1.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    @tf.custom_gradient
    def _reverse_gradient(self, x):
        """
        Internal function to define the custom gradient.
        """
        def grad(dy):
            # Reverse the gradient by multiplying with -alpha
            return -self.alpha * dy
        return x, grad

    def call(self, inputs):
        """
        Forward pass: simply returns the inputs.
        """
        return self._reverse_gradient(inputs)

    def get_config(self):
        """
        Required for saving and loading the model with custom layers.
        """
        config = super(GradientReversal, self).get_config()
        config.update({"alpha": self.alpha.numpy()})
        return config

# 2. Define the Feature Extractor Network (Conv1D)


def create_feature_extractor(input_shape):
    """
    Creates a Conv1D-based feature extractor for 1D sequence data.
    input_shape: (sequence_length, num_features)
    """
    model_input = layers.Input(
        shape=input_shape, name="feature_extractor_input")

    x = layers.Conv1D(filters=64, kernel_size=5,
                      activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=5,
                      activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=256, kernel_size=3,
                      activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Use GlobalAveragePooling for fixed-size output
    x = layers.GlobalAveragePooling1D()(x)

    # Output features for subsequent networks
    features = layers.Dense(256, activation='relu', name="features_output")(x)
    return models.Model(inputs=model_input, outputs=features, name="feature_extractor")

# 3. Define the Label Predictor Network


def create_label_predictor(num_classes):
    """
    Creates a dense network for predicting labels from extracted features.
    """
    feature_input = layers.Input(shape=(
        256,), name="label_predictor_input")  # Input shape from feature extractor
    x = layers.Dense(128, activation='relu')(feature_input)
    x = layers.Dropout(0.5)(x)
    label_output = layers.Dense(
        num_classes, activation='softmax', name="label_output")(x)
    return models.Model(inputs=feature_input, outputs=label_output, name="label_predictor")

# 4. Define the Domain Classifier Network


def create_domain_classifier():
    """
    Creates a dense network for classifying the domain (source/target) from features.
    """
    grl_input = layers.Input(
        shape=(256,), name="domain_classifier_input")  # Input shape from GRL output
    x = layers.Dense(128, activation='relu')(grl_input)
    x = layers.Dropout(0.5)(x)
    domain_output = layers.Dense(1, activation='sigmoid', name="domain_output")(
        x)  # Binary classification
    return models.Model(inputs=grl_input, outputs=domain_output, name="domain_classifier")

# 5. Combine the models into a single DANN (Domain-Adversarial Neural Network) model


def create_dann_model(input_shape, num_classes, grl_alpha=1.0):
    """
    Combines the feature extractor, label predictor, and domain classifier
    into a single DANN model.
    """
    # Input for the entire model
    model_input = layers.Input(shape=input_shape, name="main_input")

    # Feature Extractor
    feature_extractor = triplet_functions.baseTransformer(input_shape)
    features = feature_extractor(model_input)

    # Label Predictor branch
    label_predictor = create_label_predictor(num_classes)
    label_predictions = label_predictor(features)

    # Domain Classifier branch with GRL
    grl_layer = GradientReversal(alpha=grl_alpha, name="grl_layer")
    reversed_features = grl_layer(features)
    domain_classifier = create_domain_classifier()
    domain_predictions = domain_classifier(reversed_features)

    # Create the full model with two outputs
    dann_model = models.Model(
        inputs=model_input,
        outputs=[label_predictions, domain_predictions],
        name="dann_model"
    )
    return dann_model


# --- Demonstration with User's Data Loading ---
if __name__ == "__main__":
    import init_gpu
    import init_dataset
    import pandas as pd

    init_gpu.initialize_gpus()

    # Hyperparameters
    batch_size = 256
    epochs = 500
    grl_alpha = 1.0  # Strength of the gradient reversal

    # --- User's Data Loading and Preprocessing ---
    # init_gpu.initialize_gpus() # Uncomment if you have this module
    locations = ['LOC2', 'LOC3']
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    # Assuming init_dataset.get_sample is available and works as expected
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)
    # For demonstration, let's simulate get_sample if init_dataset is not available
    # In a real scenario, replace this with your actual init_dataset.get_sample call

    # IMPORTANT!: append the source data from the test set to the training set
    # This means train_df now contains source (LOC2) from original train and test,
    # and target (LOC3) from original train.
    train_df = pd.concat(
        [train_df, test_df[test_df['Location'] == locations[0]]])

    # Determine input_dim and num_classes from the loaded data
    # Exclude 'Website' (label) and 'Location' (domain) columns from features
    feature_columns = [
        col for col in train_df.columns if col not in ['Website', 'Location']]
    input_dim = len(feature_columns)

    # Assuming 'Website' contains integer labels starting from 0
    num_classes = df['Website'].nunique()

    # --- IMPORTANT CHANGE: Reshape data for Conv1D assuming input_dim is sequence_length ---
    # Reshape data for Conv1D: (num_samples, sequence_length, num_features)
    # Here, sequence_length is input_dim, and num_features is 1
    # The number of feature columns is now the sequence length
    sequence_length = input_dim
    num_features = 1  # Each element in the sequence is a single feature
    input_shape = (sequence_length, num_features)

    # Prepare Source Data
    source_df = train_df[train_df['Location'] == locations[0]]
    X_source = source_df[feature_columns].values.astype(np.float32)
    y_source = source_df['Website'].values.astype(
        np.int32)  # Labels are integer
    # Domain label for source: 0
    d_source = np.zeros((len(X_source), 1), dtype=np.float32)

    # Prepare Target Data
    target_df = train_df[train_df['Location'] == locations[1]]
    X_target = target_df[feature_columns].values.astype(np.float32)
    y_target = target_df['Website'].values.astype(
        np.int32)  # Labels are integer
    d_target = np.ones((len(X_target), 1), dtype=np.float32)

    # Reshape X_source and X_target for Conv1D input (num_samples, sequence_length, num_features)
    X_source = X_source.reshape(-1, sequence_length, num_features)
    X_target = X_target.reshape(-1, sequence_length, num_features)

    print(
        f"Source data shape: {X_source.shape}, Source labels shape: {y_source.shape}, Source domain shape: {d_source.shape}")
    print(
        f"Target data shape: {X_target.shape}, Target domain shape: {d_target.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Input dimension (original number of features): {input_dim}")
    print(f"Model input shape (sequence_length, num_features): {input_shape}")

    # Create the DANN model
    dann_model = create_dann_model(
        input_shape, num_classes, grl_alpha=grl_alpha)

    # Compile the model
    dann_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            "label_output": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            "domain_output": tf.keras.losses.BinaryCrossentropy()
        },
        loss_weights={
            "label_output": 1.0,
            "domain_output": 1.0
        },
        metrics={
            "label_output": ['accuracy'],
            "domain_output": ['accuracy']
        }
    )

    dann_model.summary()

    # --- Custom Training Loop ---
    source_dataset = tf.data.Dataset.from_tensor_slices(
        (X_source, y_source, d_source)).shuffle(buffer_size=len(X_source)).batch(batch_size)
    target_dataset = tf.data.Dataset.from_tensor_slices(
        (X_target, y_target, d_target)).shuffle(buffer_size=len(X_target)).batch(batch_size)

    source_iter = iter(source_dataset.repeat())
    target_iter = iter(target_dataset.repeat())

    optimizer = optimizers.Adam(learning_rate=0.001)

    label_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False)
    domain_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    label_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    domain_accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    @tf.function
    def train_step(source_batch, target_batch):
        X_s, y_s, d_s = source_batch
        X_t, y_t, d_t = target_batch

        X_combined_domain = tf.concat([X_s, X_t], axis=0)
        d_combined = tf.concat([d_s, d_t], axis=0)

        with tf.GradientTape() as tape:
            label_preds_s, _ = dann_model(X_s, training=True)
            label_preds_t, _ = dann_model(X_t, training=True)
            label_loss = label_loss_fn(y_s, label_preds_s) + \
                label_loss_fn(y_t, label_preds_t)

            _, domain_preds_combined = dann_model(
                X_combined_domain, training=True)
            domain_loss = domain_loss_fn(d_combined, domain_preds_combined)

            total_loss = label_loss + domain_loss

        gradients = tape.gradient(total_loss, dann_model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, dann_model.trainable_variables))

        # Update metrics
        # combine y_s and y_t for label accuracy
        label_preds_combined = tf.concat(
            [label_preds_s, label_preds_t], axis=0)
        label_accuracy_metric.update_state(
            tf.concat([y_s, y_t], axis=0), label_preds_combined)

        # label_accuracy_metric.update_state(y_s, label_preds_s)
        domain_accuracy_metric.update_state(d_combined, domain_preds_combined)

        return label_loss, domain_loss

    print("\nStarting custom training loop...")
    num_batches = min(len(X_source) // batch_size, len(X_target) // batch_size)
    if num_batches == 0:
        print("Not enough samples to form a batch. Please check your data size and batch_size.")
    else:
        for epoch in range(epochs):
            label_accuracy_metric.reset_state()
            domain_accuracy_metric.reset_state()
            total_label_loss = 0.0
            total_domain_loss = 0.0

            for i in range(num_batches):
                source_batch = next(source_iter)
                target_batch = next(target_iter)
                l_loss, d_loss = train_step(source_batch, target_batch)
                total_label_loss += l_loss
                total_domain_loss += d_loss

            avg_label_loss = total_label_loss / num_batches
            avg_domain_loss = total_domain_loss / num_batches
            label_acc = label_accuracy_metric.result()
            domain_acc = domain_accuracy_metric.result()

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Label Loss: {avg_label_loss:.4f}, Label Acc: {label_acc:.4f}, "
                  f"Domain Loss: {avg_domain_loss:.4f}, Domain Acc: {domain_acc:.4f}")

        print("\nTraining complete.")
        # You can also save the model
        # dann_model.save("dann_1d_sequence_model")
        dann_model.save(
            f"../../models-{locations[0]}-{locations[1]}/website/dann_1d_sequence_model.keras")
