import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
import numpy as np


def get_triplets_location_encoder_optimized(df, num_triplet_samples):
    # Pre-compute unique locations and websites
    unique_locations = df['Location'].unique()
    unique_websites = df['Website'].unique()

    # Create dictionaries for faster lookups
    location_to_index = {loc: i for i, loc in enumerate(unique_locations)}
    website_to_index = {web: i for i, web in enumerate(unique_websites)}

    # Convert locations and websites to integer indices
    location_indices = df['Location'].map(location_to_index).values
    website_indices = df['Website'].map(website_to_index).values

    # Pre-compute masks for each location
    location_masks = {loc: location_indices ==
                      i for i, loc in enumerate(unique_locations)}

    # Pre-compute data array
    data = df.iloc[:, 2:].to_numpy()

    # Initialize arrays for results
    n_samples = len(df) * num_triplet_samples
    anchors = np.zeros((n_samples, data.shape[1]))
    positives = np.zeros((n_samples, data.shape[1]))
    negatives = np.zeros((n_samples, data.shape[1]))
    anchor_location_labels = np.zeros(n_samples, dtype=int)

    idx = 0
    for i in range(len(df)):
        anchor_location = location_indices[i]
        anchor_website = website_indices[i]

        # Select positive samples
        positive_mask = location_masks[unique_locations[anchor_location]]
        positive_indices = np.where(positive_mask)[0]
        pos_samples = np.random.choice(
            positive_indices, num_triplet_samples, replace=True)

        # Select negative samples
        negative_mask = (~location_masks[unique_locations[anchor_location]]) & (
            website_indices == anchor_website)
        negative_indices = np.where(negative_mask)[0]
        neg_samples = np.random.choice(
            negative_indices, num_triplet_samples, replace=True)

        # Add samples to result arrays
        anchors[idx:idx+num_triplet_samples] = data[i]
        positives[idx:idx+num_triplet_samples] = data[pos_samples]
        negatives[idx:idx+num_triplet_samples] = data[neg_samples]
        anchor_location_labels[idx:idx+num_triplet_samples] = anchor_location
        idx += num_triplet_samples

    return anchors, positives, negatives


if __name__ == '__main__':
    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    num_train_samples = 1200
    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), num_train_samples)

    # get train-val set from the train set, 50 for validation set
    # training information

    num_triplet_samples = 5
    print("Generating Triplets...")
    train_anchors, train_positives, train_negatives = get_triplets_location_encoder_optimized(
        train_df, num_triplet_samples)

    print("Anchor Shape: ", train_anchors.shape)
    # Free up memory by deleting references to the DataFrames
    del train_df
    del df
    del test_df
    del train_web_samples
    del test_web_samples

    # garbage collector
    import gc
    gc.collect()

    # Training Triplet Model
    baseNetwork = 'baseGRU'
    triplet_epochs = 100

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # offline random triplet mining
        # initialize base instance
        base_instance = getattr(triplet_functions, baseNetwork)(length)

        # load existing model, continue training
        # base_instance = tf.keras.models.load_model(
        #     f'../../models/website/{locations[0]}-{locations[1]}-{baseNetwork}-epochs200-train_samples{num_train_samples}-triplet_samples{num_triplet_samples}.keras')
        # base_instance.trainable = True

        model = triplet_functions.triplet_learning(base_instance, length)
        model.compile(optimizer='adam',
                      loss=triplet_functions.triplet_loss_func)

        # Train the model with validation data and EarlyStopping
        history = model.fit(
            [train_anchors, train_positives, train_negatives],
            [train_anchors],
            epochs=triplet_epochs,
            batch_size=128,
            shuffle=True,
        )

    print("Saving model...")

    base_instance.save(
        f'../../models/location/{locations[0]}-{locations[1]}-{baseNetwork}-epochs{triplet_epochs}-train_samples{num_train_samples}-triplet_samples{num_triplet_samples}-({start_idx}, {end_idx}).keras')

    print("Location Triplet Model Training Completed!")
