import random
import triplet_functions
import init_gpu
import init_dataset
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import model_utils
from website_online_triplet import TripletSemiHardLossVectorized, TripletTrainingConfig, create_base_cnn, create_model_and_compile


def get_batched_encode(web_model, x, batch_size=16384):
    """
    Run web_model on x in smaller chunks to fit memory.
    Returns concatenated z_sample of shape (len(x), latent_dim).
    """
    z_list = []
    for i in range(0, len(x), batch_size):
        chunk = x[i:i+batch_size]
        # add only for the online triplet model
        chunk_reshaped = np.expand_dims(chunk, axis=-1)
        z_chunk = web_model.predict(chunk_reshaped)
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

    attribute = 'Resolver'
    attribute_values = ['Google', 'Cloudflare']
    train_attribute = attribute_values[0]
    test_attribute = attribute_values[1]

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv")

    length = 32
    df = df.loc[:, [attribute, 'Website', *[str(i) for i in range(length)]]]

    num_train_samples = 1200
    # get train-test set
    random.seed(RANDOM_SEED := 42)
    all_websites = range(1500)
    num_websites = 1200
    train_web_samples = random.sample(all_websites, num_websites)
    test_web_samples = list(set(all_websites) - set(train_web_samples))

    print(f"Training Websites: {train_web_samples}")
    print(f"Training Attributes: {attribute_values}")

    train_df = df[df[attribute].isin(
        attribute_values) & df["Website"].isin(train_web_samples)]
    train_df.sort_values(by=["Location"], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df = df[df[attribute].isin(attribute_values) & (df["Website"].isin(
        test_web_samples))]

    # Create a dictionary of custom objects
    custom_objects = {
        'ResidualBlock': ResidualBlock,
        "TransformerEncoderBlock": triplet_functions.TransformerEncoderBlock,
        "TripletSemiHardLossVectorized": TripletSemiHardLossVectorized,
    }

    # 1) Re‑instantiate the model graph
    config = TripletTrainingConfig(feature_length=32,
                                   num_train_samples=1200,
                                   batch_size=128,
                                   epochs=1000,
                                   learning_rate=1e-4,
                                   weight_decay=1e-5,
                                   margin=0.2,
                                   patience=50,
                                   validation_split=0.2,
                                   base_network_name='baseCNN')

    base_net = create_base_cnn(config.feature_length, embedding_dim=64)
    web_model = create_model_and_compile(base_net, config)
    web_model.load_weights(
        f"../../models/website/Lausanne-Leuven-Singapore-baseCNN-online_semi_hard_AdamW-epochs1000-train_samples1200-batch128_best.keras"
    )

    # Load the model with custom objects
    # tf.keras.config.enable_unsafe_deserialization()
    # web_model = tf.keras.models.load_model(
    #     f"../../models/website/Lausanne-Leuven-Singapore-baseCNN-online_semi_hard_AdamW-epochs1000-train_samples1200-batch128_best.keras",
    #     custom_objects=custom_objects
    # )

    le = LabelEncoder()
    X_train = df[df['Location'] == train_attribute].drop(
        ['Location', 'Website'], axis=1)
    X_test = df[df['Location'] == test_attribute].drop(
        ['Location', 'Website'], axis=1)
    y_train = df[df['Location'] == train_attribute]['Website']
    y_test = df[df['Location'] == test_attribute]['Website']

    y_test = le.fit_transform(y_test)
    y_train = le.fit_transform(y_train)

    print("Evaluating the model...")
    print("Without Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        X_train, y_train, X_test, y_test, model)

    print("With Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        get_batched_encode(web_model, X_train), y_train, get_batched_encode(web_model, X_test), y_test, model)

    print("Evaluating the model on training on source data...")
    source_df = df[df[attribute] == train_attribute]
    target_df = df[df[attribute] == test_attribute]
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
    X_test = test_df[test_df[attribute] == train_attribute
                     ].iloc[:, 2:].to_numpy().astype(np.float32)
    y_test = test_df[test_df[attribute] == test_attribute
                     ]['Website'].to_numpy().astype(np.int32)
    y_test = le.transform(y_test)

    print("With Embedding:")
    model = KNeighborsClassifier(n_neighbors=10)
    classification.evaluate_classification_model(
        get_batched_encode(web_model, X_train), y_train, get_batched_encode(web_model, X_test), y_test, model)

    print("Done evaluating the model.")
