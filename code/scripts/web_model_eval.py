import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import model_utils


def get_batched_encode(web_model, x, batch_size=2048):
    """
    Run web_model on x in smaller chunks to fit memory.
    Returns concatenated z_sample of shape (len(x), latent_dim).
    """
    z_list = []
    for i in range(0, len(x), batch_size):
        chunk = x[i:i+batch_size]
        z_chunk = web_model(chunk)
        z_list.append(z_chunk)
    return np.concatenate(z_list, axis=0)


if __name__ == '__main__':
    init_gpu.initialize_gpus()
    from sklearn.neighbors import KNeighborsClassifier
    from triplet_functions import ResidualBlock
    from downstream_classification import generate_synthetic_data
    from train_vae import ConvVAE_BatchNorm, Sampling
    from hyperplane import get_hyperplane
    import classification

    locations = ['LOC2', 'LOC3']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    num_train_samples = 1200
    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), num_train_samples)

    # Create a dictionary of custom objects
    custom_objects = {
        'ResidualBlock': ResidualBlock
    }

    # Load the model with custom objects
    web_model = tf.keras.models.load_model(
        f"../../models-LOC2-LOC3/website/{locations[0]}-{locations[1]}-baseCNN-epochs500-train_samples1200-triplet_samples5-domain_invariant-l0.1.keras",
        custom_objects=custom_objects
    )

    X_train, y_train, X_test, y_test, le = classification.preprocess_data_for_web_classification(
        test_df, locations[0], locations[1])

    print("Evaluating the model...")
    print("Without Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        X_train, y_train, X_test, y_test, model)

    print("With Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        web_model(X_train), y_train, web_model(X_test), y_test, model)

    print("Evaluating the model on training on source data...")
    source_df = df[df['Location'] == locations[0]]
    target_df = df[df['Location'] == locations[1]]
    X_train = source_df.iloc[:, 2:].to_numpy().astype(np.float32)
    y_train = source_df['Website'].to_numpy().astype(np.int32)
    X_test = target_df.iloc[:, 2:].to_numpy().astype(np.float32)
    y_test = target_df['Website'].to_numpy().astype(np.int32)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    print("Without Embedding:")

    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        X_train, y_train, X_test, y_test, model)
    print("With Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        get_batched_encode(web_model, X_train), y_train, get_batched_encode(web_model, X_test), y_test, model)

    print("Test only on the unseen websites in the target location.")
    X_test = test_df[test_df['Location'] == locations[1]
                     ].iloc[:, 2:].to_numpy().astype(np.float32)
    y_test = test_df[test_df['Location'] == locations[1]
                     ]['Website'].to_numpy().astype(np.int32)
    y_test = le.transform(y_test)

    print("With Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        get_batched_encode(web_model, X_train), y_train, get_batched_encode(web_model, X_test), y_test, model)

    print("Done evaluating the model.")
