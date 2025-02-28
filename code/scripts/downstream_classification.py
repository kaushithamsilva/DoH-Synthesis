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


def generate_synthetic_data(source_df, w):
    synthetic_df = pd.DataFrame(columns=source_df.columns)

    print("Getting z embeddings...")
    source_z = get_z_embeddings(source_df.iloc[:, 2:].to_numpy())

    # Define the traversal factors
    traversal_factors = np.linspace(-15.0, -5, 100)

    for factor in traversal_factors:
        print(f"Traversing z embeddings with factor {factor}...")
        z_traversed = source_z + factor * 2.0 * w

        print("Decoding z embeddings...")
        decoded_data = get_decoded_data(z_traversed)

        temp_synthetic_df = source_df.copy()
        temp_synthetic_df.iloc[:, 2:] = decoded_data

        synthetic_df = pd.concat(
            [synthetic_df, temp_synthetic_df], ignore_index=True)

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
    synthetic_df = generate_synthetic_data(source_df, w)

    # Prepare the training and test datasets
    le = LabelEncoder()
    X_train = synthetic_df.iloc[:, 2:]
    y_train = le.fit_transform(synthetic_df.Website)
    X_test = target_df.iloc[:, 2:]
    y_test = le.transform(target_df.Website)

    model = build_classifier(
        input_dim=length, hidden_dim=96, num_classes=len(test_web_samples))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, batch_size=32, epochs=20, shuffle=True)

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
