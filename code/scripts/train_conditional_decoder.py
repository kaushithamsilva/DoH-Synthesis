
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Reshape, GRU
from tensorflow.keras.optimizers import Adam
import init_gpu as init_gpu
import init_dataset as init_dataset
from triplet_functions import n_neurons
import pandas as pd
import tensorflow as tf

import numpy as np


def get_web_embeddings(df, location, web_model):
    embeddings = []
    chunk_size = 10000
    for i in range(0, 240000, chunk_size):
        chunk = filter_and_sort_data(df[i:i+chunk_size], location)

        # Pad the chunk if it's smaller than chunk_size
        if len(chunk) < chunk_size:
            chunk = chunk.reindex(range(chunk_size), fill_value=0)

        transformed_chunk = web_model(chunk)
        embeddings.append(transformed_chunk)

    return np.vstack(embeddings)


def get_data_for_decoder_preserver_locality(df, locations):
    input_data_list = []  # 240,000 * 2 * 2
    output_data_list = []

    for location in locations:
        web_embeddings = get_web_embeddings(df, location, web_model)

        # Preprocess input data
        location_labels = np.tile(
            location_to_onehot_dict[location], (web_embeddings.shape[0], 1))
        input_embedding_with_condition = np.hstack(
            (web_embeddings, location_labels))

        # Preprocess output data
        output_traces = filter_and_sort_data(df, location).to_numpy()

        # Append both input and output to dataset lists
        input_data_list.append(input_embedding_with_condition)
        output_data_list.append(output_traces)

    # Concatenate all data
    input_data = np.vstack(input_data_list)
    output_data = np.vstack(output_data_list)

    return input_data, output_data


def get_data_for_decoder(df, locations):
    input_data_list = []  # 240,000 * 2 * 2
    output_data_list = []

    for input_location in locations:
        # Get web embedding from the input location
        web_embeddings = get_web_embeddings(df, input_location, web_model)

        for output_location in locations:
            # Preprocess input data
            location_labels = np.tile(
                location_to_onehot_dict[output_location], (web_embeddings.shape[0], 1))
            input_embedding_with_condition = np.hstack(
                (web_embeddings, location_labels))

            # Preprocess output data
            output_traces = filter_and_sort_data(
                df, output_location).to_numpy()

            # Append both input and output to dataset lists
            input_data_list.append(input_embedding_with_condition)
            output_data_list.append(output_traces)

    # Concatenate all data
    input_data = np.vstack(input_data_list)
    output_data = np.vstack(output_data_list)

    return input_data, output_data


def filter_and_sort_data(df, location_label):
    """
    Filter the dataframe by location label, sort by Website label, and drop specified columns.
    """
    return (df[df['Location'] == location_label]
            .sort_values(by=['Website'])
            .iloc[:, 2:]
            .reset_index(drop=True))


def location_to_onehot(select_location, locations):
    if select_location not in locations:
        raise ValueError(
            f"'{select_location}' is not in the list of locations")

    return [1 if location == select_location else 0 for location in locations]


if __name__ == '__main__':
    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    web_model = tf.keras.models.load_model(
        f"../../models/website/{locations[0]}-{locations[1]}-baseGRU-epochs200-train_samples1200-triplet_samples5.keras")
    location_to_onehot_dict = {location: location_to_onehot(
        location, locations) for location in locations}

    input_train_data, output_train_data = get_data_for_decoder_preserver_locality(
        train_df, locations)

    latent_length = n_neurons + len(locations)
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=(latent_length, )),
        tf.keras.layers.Dense(64, activation="tanh"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.RepeatVector(1),  # Prepare for GRU layers
        tf.keras.layers.GRU(
            length, return_sequences=False, activation="tanh"),
        # tf.keras.layers.GRU(
        #     length, return_sequences=False, activation="tanh"),
        tf.keras.layers.Dense(
            length, activation='linear')  # Output layer
    ])
    decoder.compile(optimizer=Adam(), loss='mse', )

    # Fit the model
    decoder.fit(input_train_data, output_train_data,
                epochs=1000, batch_size=32, shuffle=True)
    decoder.save(
        f'../../models/decoder/LOC1-LOC2-Conditional-Decoder-e1000-locality-preserved.keras')
