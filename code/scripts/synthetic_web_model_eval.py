import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
import model_utils

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
        source_df, w, b, vae_model, n_samples=100, n_interpolations=2, n_pairs=2)

    synthetic_df['Location'] = target_location

    # Create a dictionary of custom objects
    custom_objects = {
        'ResidualBlock': ResidualBlock
    }

    # Load the model with custom objects
    web_model = tf.keras.models.load_model(
        "../../models-LOC2-LOC3/website/LOC2-LOC3-baseCNN-epochs500-train_samples1200-triplet_samples5-domain_invariant-l0.1.keras",
        custom_objects=custom_objects
    )

    X_train, y_train, X_test, y_test, le = classification.preprocess_data_for_web_classification(
        test_df, locations[0], locations[1])

    print("Evaluating the model...")
    print("KNN Classifier trained on actual source data")
    print("\tWithout Embedding:")
    model = KNeighborsClassifier(n_neighbors=1)
    classification.evaluate_classification_model(
        X_train, y_train, X_test, y_test, model)

    print("\tWith Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        web_model(X_train), y_train, web_model(X_test), y_test, model)

    print("KNN Classifier trained on target synthetic data")

    X_train = synthetic_df.iloc[:, 2:]
    y_train = le.transform(synthetic_df['Website'])

    print("\tWithout Embedding:")
    model = KNeighborsClassifier(n_neighbors=1)
    classification.evaluate_classification_model(
        X_train, y_train, X_test, y_test, model)

    print("\With Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        web_model(X_train), y_train, web_model(X_test), y_test, model)
