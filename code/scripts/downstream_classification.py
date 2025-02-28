# Downstream classification tasks
# Classification: Trained on synthetic data, test on real data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import random
from tensorflow import keras
import tensorflow as tf


class ResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv1D(
            filters, kernel_size, activation="relu", padding="same")
        self.conv2 = keras.layers.Conv1D(
            filters, kernel_size, activation=None, padding="same")
        self.activation = keras.layers.Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return self.activation(x + inputs)  # Adding the residual connection


def build_classifier(input_dim, hidden_dim, num_classes):
    base_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim, 1)),
        keras.layers.Conv1D(hidden_dim, kernel_size=7,
                            strides=2, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        ResidualBlock(hidden_dim, kernel_size=5),
        keras.layers.Conv1D(hidden_dim * 2, kernel_size=3,
                            strides=2, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        ResidualBlock(hidden_dim * 2, kernel_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
    ])

    classifier = keras.Sequential([
        base_model,
        keras.layers.Dense(hidden_dim * 2, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes)  # No softmax for logits
    ])

    return classifier


def get_z_embeddings(data, vae_model):
    embeddings = []
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        _, _, transformed_chunk = vae_model.encode(chunk)
        embeddings.append(transformed_chunk)

    return np.vstack(embeddings)


def get_decoded_data(data, vae_model):
    decoded_data = []
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        decoded_chunk = vae_model.decode(chunk)
        decoded_data.append(decoded_chunk)

    return np.vstack(decoded_data)


def get_interpolated_df(z_embeddings, df, vae_model, n_interpolations, n_pairs):
    """
    Create a DataFrame of interpolated latent embeddings.

    Parameters:
    - z_embeddings: array-like of shape (N, d), latent embeddings in the same order as rows in df.
    - df: original DataFrame with columns "website" and "location" (plus others).
    - vae_model: VAE model (not used in interpolation here, but kept for interface consistency)
    - n_interpolations: integer, number of interpolated samples to generate between each consecutive pair.

    Returns:
    - interpolated_df: DataFrame with columns: website, location, 1, 2, ..., d
    """
    all_embeddings = []
    all_websites = []
    all_locations = []

    # Precompute the interpolation coefficients (alphas) as a vector.
    # For example, if n_interpolations=1, alphas will be [0.5] (midpoint).
    # shape (n_interpolations,)
    alphas = np.linspace(0, 1, n_interpolations + 2)[1:-1]

    # Group the DataFrame by website and location.
    groups = df.groupby(['Website', 'Location'])

    # for consecutive pairs
    """
    for (website, location), group in groups:
        # Get indices as a NumPy array (ensuring integer type)
        indices = group.index.to_numpy()
        # Need at least two embeddings to perform interpolation
        if len(indices) < 2:
            continue

        # Assuming z_embeddings is a numpy array, select embeddings for this group.
        group_embeddings = z_embeddings[indices]  # shape: (n, d)

        # Compute consecutive pairs: (start, end)
        start = group_embeddings[:-1]  # shape: (n-1, d)
        end = group_embeddings[1:]     # shape: (n-1, d)

        # Use broadcasting to compute interpolated embeddings.
        # start[:, None, :] has shape (n-1, 1, d), and alphas reshaped to (1, n_interpolations, 1)
        # This yields an array of shape (n-1, n_interpolations, d).
        interp = start[:, None, :] * (1 - alphas)[None, :,
                                                  None] + end[:, None, :] * alphas[None, :, None]
        # Reshape to a 2D array where each row is one interpolated embedding.
        # shape: ((n-1)*n_interpolations, d)
        interp = interp.reshape(-1, group_embeddings.shape[1])

        # Create corresponding website and location arrays.
        num_interp = interp.shape[0]
        websites = np.full(num_interp, website)
        locations = np.full(num_interp, location)

        # Append the results.
        all_embeddings.append(interp)
        all_websites.append(websites)
        all_locations.append(locations)
    """

    # for randomly chosen n-pairs
    for (website, location), group in groups:
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue  # Skip groups with only 1 sample

        group_embeddings = z_embeddings[indices]  # Shape: (num_samples, d)

        for i in range(len(group_embeddings)):
            # Randomly select `n_pairs` distinct embeddings to interpolate with (excluding self)
            possible_indices = list(range(len(group_embeddings)))
            possible_indices.remove(i)  # Avoid choosing itself
            chosen_indices = np.random.choice(possible_indices, min(
                n_pairs, len(possible_indices)), replace=False)

            for j in chosen_indices:
                start, end = group_embeddings[i], group_embeddings[j]

                # Compute interpolations
                interp = start[None, :] * \
                    (1 - alphas)[:, None] + end[None, :] * alphas[:, None]

                all_embeddings.append(interp)
                all_websites.append(np.full(len(alphas), website))
                all_locations.append(np.full(len(alphas), location))

    # If no interpolated samples were generated, return an empty DataFrame.
    if not all_embeddings:
        return pd.DataFrame()

    # Concatenate results from all groups.
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_websites = np.concatenate(all_websites, axis=0)
    all_locations = np.concatenate(all_locations, axis=0)

    # Create column names: 'website', 'location', then "1", "2", ..., "d"
    embedding_dim = all_embeddings.shape[1]
    columns = ['Website', 'Location'] + \
        [str(i+1) for i in range(embedding_dim)]

    # Combine website, location and embeddings into one DataFrame.
    # Note: We use np.column_stack to combine object-type data with numerical arrays.
    data = np.column_stack((all_websites, all_locations, all_embeddings))
    interpolated_df = pd.DataFrame(data, columns=columns)
    interpolated_df['Website'] = interpolated_df['Website'].astype(int)

    return interpolated_df


def traverse(z, factor, w):
    return z + factor * w


def generate_synthetic_data(source_df, w, vae_model):
    synthetic_dfs = []

    print("Getting z embeddings...")
    source_traces = source_df.iloc[:, 2:].to_numpy()
    source_z = get_z_embeddings(source_traces, vae_model)

    # Generate interpolated data; note that interpolated_df already has 'Website' and 'Location' as its first two columns.
    interpolated_df = get_interpolated_df(
        source_z, source_df, vae_model, n_interpolations=2, n_pairs=5)
    interpolated_z = interpolated_df.iloc[:, 2:].to_numpy()

    # Define the traversal factors
    traversal_factors = np.linspace(-30.0, -10.0, 2)

    # Helper function: traverse, decode, and build a DataFrame including Website and Location.
    def decode_and_create_df(z_values, websites, locations):
        decoded_data = get_decoded_data(z_values, vae_model)
        # Create DataFrame using the decoded data and assign the same column names as in the original trace columns.
        df_decoded = pd.DataFrame(decoded_data, columns=source_df.columns[2:])
        # Insert Website and Location as the first two columns.
        df_decoded.insert(0, 'Location', locations)
        df_decoded.insert(0, 'Website', websites)
        return df_decoded

    # Loop over each traversal factor
    for factor in traversal_factors:
        print(f"Traversing z embeddings with factor {factor}...")

        # Process the original (non-interpolated) data
        z_traversed_orig = traverse(source_z, factor, w)
        df_orig = decode_and_create_df(z_traversed_orig,
                                       source_df['Website'].values,
                                       source_df['Location'].values)
        synthetic_dfs.append(df_orig)

        # Process the interpolated data
        z_traversed_interp = traverse(interpolated_z, factor, w)
        df_interp = decode_and_create_df(z_traversed_interp,
                                         interpolated_df['Website'].values,
                                         interpolated_df['Location'].values)
        synthetic_dfs.append(df_interp)

    # Concatenate all the synthetic DataFrames at once
    synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)
    return synthetic_df


if __name__ == '__main__':
    import init_dataset
    import init_gpu
    from hyperplane import get_hyperplane
    from train_vae import VAE, Sampling, ConvVAE, ConvVAE_BatchNorm
    init_gpu.initialize_gpus()

    print("Loading Dataset...")
    # load the dataset
    locations = ['LOC2', 'LOC3']
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)
    source_location, target_location = locations

    # data preprocessing for source, real target, and synthetic data
    target_df = test_df[test_df['Location'] == target_location]
    target_df.sort_values(by=['Website'], inplace=True)
    target_df.reset_index(drop=True, inplace=True)
    target_df.head(20)

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
    synthetic_df = generate_synthetic_data(source_df, w, vae_model)

    # Prepare the training and test datasets
    le = LabelEncoder()
    X_train = synthetic_df.iloc[:, 2:]
    y_train = le.fit_transform(synthetic_df.Website)
    X_test = target_df.iloc[:, 2:]
    y_test = le.transform(target_df.Website)

    # GRY model
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(length, 1)),
        keras.layers.GRU(96),
        keras.layers.Dense(len(test_web_samples))
    ])

    # model = build_classifier(
    #    input_dim=length, hidden_dim=96, num_classes=len(test_web_samples))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, batch_size=256, epochs=5, shuffle=True)

    # Get logits from model prediction
    logits = model.predict(X_test)

    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(logits).numpy()

    # Get predicted class by selecting the class with highest probability
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Use weighted average for imbalanced data
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")
