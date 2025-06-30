import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.losses import TripletSemiHardLoss


def create_tf_dataset(df, batch_size):
    """
    Creates a TensorFlow dataset from a DataFrame, yielding features and website labels.
    """
    # Extract features (data after 'Location' and 'Website') and website labels
    features = df.iloc[:, 2:].values.astype(np.float32)
    website_labels = df['Website'].values

    # Create a tf.data.Dataset from features and labels
    dataset = tf.data.Dataset.from_tensor_slices((features, website_labels))

    # Shuffle, batch, and prefetch for performance
    dataset = dataset.shuffle(buffer_size=len(df)).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Define the Triplet Semi-Hard Loss function (standard TensorFlow implementation)
# This loss expects embeddings and labels as input.
# The margin should be chosen based on your dataset and desired separation.
# A common starting point is 0.2 or 1.0.
triplet_loss_metric = TripletSemiHardLoss(margin=0.2)


class TripletOnlineMiningModel(models.Model):
    """
    Custom Keras Model for online triplet mining.
    It takes a base network and computes embeddings, then applies the
    TripletSemiHardLoss.
    """

    def __init__(self, base_network, **kwargs):
        super(TripletOnlineMiningModel, self).__init__(**kwargs)
        self.base_network = base_network
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        # The inputs here will be the features from the dataset batch
        return self.base_network(inputs)

    def train_step(self, data):
        # 'data' will be a tuple: (features, website_labels) from our dataset
        features, website_labels = data

        with tf.GradientTape() as tape:
            # Get embeddings for the entire batch
            embeddings = self.base_network(features, training=True)
            # Calculate the triplet semi-hard loss
            # tf.losses.TripletSemiHardLoss internally handles the mining
            # based on embeddings and labels within the batch.
            loss = triplet_loss_metric(
                y_true=website_labels, y_pred=embeddings)

        # Apply gradients
        gradients = tape.gradient(loss, self.base_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.base_network.trainable_weights))

        # Update metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so that .reset_states() can be called automatically
        return [self.loss_tracker]


if __name__ == '__main__':
    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2', 'LOC3']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv"
    )

    length = len(df.columns) - 2  # subtract the two label columns

    num_train_samples = 1200
    # get train-test set (assuming this splits the original df into train_df and test_df)
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), num_train_samples)

    print("Creating TensorFlow Dataset for Training...")
    # Batch size for training. This batch will be used for online triplet mining.
    batch_size = 128
    train_dataset = create_tf_dataset(train_df, batch_size=batch_size)

    # Free up memory by deleting references to the DataFrames
    del df
    del train_df
    del test_df
    import gc
    gc.collect()

    # Training Triplet Model
    baseNetwork_name = 'baseCNN'
    triplet_epochs = 1000

    # Initialize base instance.
    base_instance = getattr(triplet_functions, baseNetwork_name)(length)

    # Create the online mining model
    model = TripletOnlineMiningModel(base_instance)

    # Compile the model with AdamW optimizer
    # You might need to tune the learning_rate and weight_decay
    learning_rate = 1e-4  # Common starting point
    weight_decay = 1e-5   # Common starting point for regularization
    model.compile(optimizer=AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay))

    print("Training Triplet Model with Online Semi-Hard Mining using AdamW...")
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=triplet_epochs,
    )

    print("Saving base network model...")
    # Save only the base network which outputs the embeddings
    base_instance.save(
        f'../../models/website/{locations[0]}-{locations[1]}-{baseNetwork_name}-online_semi_hard_AdamW-epochs{triplet_epochs}-train_samples{num_train_samples}-batch{batch_size}.keras'
    )

    print("Website Triplet Model Training Completed!")
