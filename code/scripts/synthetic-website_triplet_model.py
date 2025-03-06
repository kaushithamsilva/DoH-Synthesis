import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
import model_utils


def get_triplets_website_encoder(df, num_triplet_samples):
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

    idx = 0
    for i in range(len(df)):
        # print(f"Processing {i} out of {len(df)}")
        anchor_location = location_indices[i]
        anchor_website = website_indices[i]

        # Select negative samples
        negative_mask = (website_indices != anchor_website)
        negative_indices = np.where(negative_mask)[0]
        neg_samples = np.random.choice(
            negative_indices, num_triplet_samples, replace=True)

        # Select positive samples
        positive_mask = (website_indices == anchor_website) & (
            location_indices != anchor_location)
        positive_indices = np.where(positive_mask)[0]
        pos_samples = np.random.choice(
            positive_indices, num_triplet_samples, replace=True)

        # Add samples to result arrays
        anchors[idx:idx+num_triplet_samples] = data[i]
        positives[idx:idx+num_triplet_samples] = data[pos_samples]
        negatives[idx:idx+num_triplet_samples] = data[neg_samples]
        anchor_website_labels[idx:idx+num_triplet_samples] = anchor_website

        idx += num_triplet_samples

    return anchor_website_labels.reshape(-1, 1), anchors, positives, negatives


if __name__ == '__main__':
    init_gpu.initialize_gpus()
    from downstream_classification import generate_synthetic_data
    from train_vae import ConvVAE_BatchNorm, Sampling
    from hyperplane import get_hyperplane

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

    source_location, target_location = locations

    # data preprocessing for source, real target, and synthetic data
    source_df = test_df[test_df['Location'] == source_location]
    source_df.sort_values(by=['Website'], inplace=True)
    source_df.reset_index(drop=True, inplace=True)

    # load models
    latent_dim = 96
    vae_model = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/vae-e1000-mse1-kl0.0001-cl1.0-ldim96-hdim128.keras", custom_objects={
                                           'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    domain_discriminator = tf.keras.models.load_model(
        f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/domain-discriminator-e1000.keras")

    # get the hyperplane
    w, b = get_hyperplane(domain_discriminator)
    synthetic_df = generate_synthetic_data(
        source_df, w, b, vae_model, n_samples=1, n_interpolations=2, n_pairs=1)
    synthetic_df['Location'] = target_location
    combined_df = pd.concat([train_df, synthetic_df], ignore_index=True)

    # get train-val set from the train set, 50 for validation set
    # training information

    num_triplet_samples = 5
    print("Generating Triplets...")
    train_anchor_labels, train_anchors, train_positives, train_negatives = get_triplets_website_encoder(
        combined_df, num_triplet_samples)

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
    triplet_epochs = 3

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        # offline random triplet mining
        # initialize base instance
        base_instance = getattr(triplet_functions, baseNetwork)(length)

        # load existing model, continue training
        # base_instance = tf.keras.models.load_model(
        #     f'../../models/website/{locations[0]}-{locations[1]}-{baseNetwork}-epochs200-train_samples{num_train_samples}-triplet_samples5.keras')
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

    model_utils.save_model(
        base_instance, f'../../models-{locations[0]}-{locations[1]}/website', f'synthetic-{locations[0]}-{locations[1]}-{baseNetwork}-epochs{triplet_epochs}-train_samples{num_train_samples}-triplet_samples{num_triplet_samples}')
    print("Website Triplet Model Training Completed!")
