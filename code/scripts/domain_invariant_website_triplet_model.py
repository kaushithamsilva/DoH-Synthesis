import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GRU, RepeatVector, Input
from triplet_functions import n_neurons


class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_value=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_value = lambda_value

    def call(self, x):
        @tf.custom_gradient
        def grad_reverse(x):
            def custom_grad(dy):
                return -self.lambda_value * dy
            return x, custom_grad
        return grad_reverse(x)


def triplet_loss_func_with_domain_misclassification(y_true, y_pred, alpha=0.2):
    global n_neurons
    """
    Implementation of the triplet loss + domain classification loss
    """
    # Split the predictions into triplet embeddings and domain predictions
    anchor = y_pred[:, 0:n_neurons]
    positive = y_pred[:, n_neurons:2*n_neurons]
    negative = y_pred[:, 2*n_neurons:3*n_neurons]
    domain_pred = y_pred[:, 3*n_neurons:]  # Domain classifier output

    # Triplet loss
    pos_dist = tf.math.reduce_sum(tf.math.square(anchor-positive), axis=1)
    neg_dist = tf.math.reduce_sum(tf.math.square(anchor-negative), axis=1)
    triplet_loss = tf.reduce_mean(
        tf.math.maximum(pos_dist-neg_dist+alpha, 0.0))

    # Domain classification loss (binary cross-entropy)
    domain_true = y_true  # Last column contains domain labels
    domain_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(domain_true, domain_pred))

    # Log the individual losses
    # Log the individual losses into the logs dictionary

    # Total loss is triplet loss + domain loss
    # The GRL will handle the gradient reversal for domain_loss
    total_loss = triplet_loss + domain_loss
    return total_loss


def triplet_learning_domain_invariant(base_network, domain_classifier, length, lambda_param=1.0):
    positive_example = Input(shape=(length, 1))
    negative_example = Input(shape=(length, 1))
    anchor_example = Input(shape=(length, 1))

    # Get embeddings
    positive_embedding = base_network(positive_example)
    negative_embedding = base_network(negative_example)
    anchor_embedding = base_network(anchor_example)

    # Add gradient reversal layer before domain classifier
    grl = GradientReversalLayer(lambda_param)
    reversed_anchor_features = grl(anchor_embedding)

    # Get domain predictions
    domain_predictions = domain_classifier(reversed_anchor_features)

    # Concatenate triplet embeddings and domain predictions
    merged_output = tf.keras.layers.concatenate(
        [anchor_embedding, positive_embedding, negative_embedding, domain_predictions])

    model = Model(
        inputs=[anchor_example, positive_example, negative_example],
        outputs=merged_output)
    return model


def get_triplets_website_encoder_optimized(df, num_triplet_samples):
    # Pre-compute unique locations and websites
    unique_locations = df['Location'].unique()
    unique_websites = df['Website'].unique()

    # Create dictionaries for faster lookups
    location_to_index = {loc: i for i, loc in enumerate(unique_locations)}
    website_to_index = {web: i for i, web in enumerate(unique_websites)}

    # Convert locations and websites to integer indices
    location_indices = df['Location'].map(location_to_index).values
    website_indices = df['Website'].map(website_to_index).values

    # Pre-compute masks for each website
    website_masks = {web: website_indices ==
                     i for i, web in enumerate(unique_websites)}

    # Pre-compute data array
    data = df.iloc[:, 2:].to_numpy()

    # Initialize arrays for results
    n_samples = len(df) * num_triplet_samples
    anchors = np.zeros((n_samples, data.shape[1]))
    positives = np.zeros((n_samples, data.shape[1]))
    negatives = np.zeros((n_samples, data.shape[1]))
    anchor_website_labels = np.zeros(n_samples, dtype=int)
    anchor_location_labels = np.zeros(n_samples, dtype=int)  # Added this line

    idx = 0
    for i in range(len(df)):
        anchor_location = location_indices[i]
        anchor_website = website_indices[i]

        # Select negative samples
        negative_mask = (location_indices != anchor_location) & (
            website_indices != anchor_website)
        negative_indices = np.where(negative_mask)[0]
        neg_samples = np.random.choice(
            negative_indices, num_triplet_samples, replace=True)

        # Select positive samples
        positive_mask = website_masks[unique_websites[anchor_website]]
        positive_indices = np.where(positive_mask)[0]
        pos_samples = np.random.choice(
            positive_indices, num_triplet_samples, replace=True)

        # Add samples to result arrays
        anchors[idx:idx+num_triplet_samples] = data[i]
        positives[idx:idx+num_triplet_samples] = data[pos_samples]
        negatives[idx:idx+num_triplet_samples] = data[neg_samples]
        anchor_website_labels[idx:idx+num_triplet_samples] = anchor_website
        anchor_location_labels[idx:idx +
                               num_triplet_samples] = float(anchor_location)

        idx += num_triplet_samples

    return (anchor_website_labels.reshape(-1, 1),
            anchor_location_labels.reshape(-1, 1),
            anchors,
            positives,
            negatives)


def classifier_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    init_gpu.initialize_gpus()

    locations = ['LOC2', 'LOC3']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    # # limit the length
    # start_idx, end_idx = 20, 40
    # df = df.loc[:, ['Location', 'Website', *
    #                 [str(i) for i in range(start_idx, end_idx)]]]
    # print(f'Columns Considered: {df.columns}')

    length = len(df.columns) - 2  # subtract the two label columns

    num_train_samples = 1200
    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), num_train_samples)

    # seen_df, unseen_df = init_dataset.get_seen_unseen_df(
    #     train_df, test_df, 'LOC2', 'LOC3')
    # get train-val set from the train set, 50 for validation set
    # training information

    num_triplet_samples = 5
    print("Generating Triplets...")
    train_anchor_labels, train_anchor_location_labels, train_anchors, train_positives, train_negatives = get_triplets_website_encoder_optimized(
        train_df, num_triplet_samples)

    print("Anchor Shape: ", train_anchors.shape)
    # Free up memory by deleting references to the DataFrames
    del train_df
    del df
    del train_anchor_labels
    del test_df
    del train_web_samples
    del test_web_samples

    # garbage collector
    import gc
    gc.collect()

    # Training Triplet Model
    baseNetwork = 'baseCNN'
    triplet_epochs = 500

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # offline random triplet mining
        # initialize base instance
        base_instance = getattr(triplet_functions, baseNetwork)(length)

        # initialize domain classifier network
        domain_classifier = classifier_network(n_neurons)

        # load existing model for both triplet embedder and classifier for a head start
        # base_instance = tf.keras.models.load_model(
        #     f'../../models/website/{locations[0]}-{locations[1]}-{baseNetwork}-epochs100-train_samples{1200}-triplet_samples5-domain_invariant-l1.keras')
        # base_instance.trainable = True

        # domain_classifier = tf.keras.models.load_model(
        #     f'../../models/classification/location/triplet_classifier.keras'
        # )

        # domain_classifier.trainable = True

        lambda_parameter = 0.1
        # Create combined model with gradient reversal
        model = triplet_learning_domain_invariant(
            base_instance, domain_classifier, length, lambda_param=lambda_parameter)
        model.compile(optimizer='adam',
                      loss=triplet_loss_func_with_domain_misclassification)
        # Train the model
        history = model.fit(
            [train_anchors, train_positives, train_negatives],
            # Concatenate anchor embeddings with location labels
            train_anchor_location_labels,
            epochs=triplet_epochs,
            batch_size=128,
            shuffle=True,
        )

    print("Saving model...")

    base_instance.save(
        f'../../models-LOC2-LOC3/website/{locations[0]}-{locations[1]}-{baseNetwork}-epochs{triplet_epochs}-train_samples{num_train_samples}-triplet_samples{num_triplet_samples}-domain_invariant-l{lambda_parameter}.keras')

    print("Website Triplet Model Training Completed!")
