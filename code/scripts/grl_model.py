import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import triplet_functions  # for baseCNN


class GradientReversalLayer(keras.layers.Layer):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def call(self, x):
        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse_gradient(x)


def build_grl_model(input_dim, num_classes, num_locations, grl_lambda=1.0):
    # Feature extractor
    inputs = keras.Input(shape=(input_dim,))
    features = triplet_functions.baseCNN(input_dim)(inputs)

    # Label classifier head
    label_preds = layers.Dense(
        num_classes, activation='softmax', name='label_classifier')(features)

    # Domain classifier head with GRL
    x_grl = GradientReversalLayer(lambda_=grl_lambda)(features)
    x = layers.Dense(64, activation='relu')(x_grl)
    domain_preds = layers.Dense(
        num_locations, activation='softmax', name='domain_classifier')(x)

    model = keras.Model(inputs=inputs,
                        outputs=[label_preds, domain_preds],
                        name='feature_extractor_grl')
    return model


if __name__ == '__main__':
    import init_gpu
    import init_dataset
    import pandas as pd
    import numpy as np

    # Example usage
    input_dim = 128  # e.g., feature vector length
    num_classes = 1500
    num_locations = 2
    grl_lambda = 0.5

    model = build_grl_model(input_dim, num_classes, num_locations, grl_lambda)
    model.compile(
        optimizer='adam',
        loss={
            'label_classifier': 'sparse_categorical_crossentropy',
            'domain_classifier': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'label_classifier': 1.0,
            'domain_classifier': 1.0
        },
        metrics={
            'label_classifier': 'accuracy',
            'domain_classifier': 'accuracy'
        }
    )
    model.summary()

    init_gpu.initialize_gpus()
    locations = ['LOC2', 'LOC3']
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)
    input_dim = train_df.shape[1] - 2

    # IMPORTANT!: append the source data from the test set to the training set
    train_df = pd.concat(
        [train_df, test_df[test_df['Location'] == locations[0]]])

    X = train_df.iloc[:, 2:].to_numpy().astype(np.float32)
    y = train_df['Website'].to_numpy().astype(int)
    d = train_df['Location'].apply(
        lambda x: 0 if x == locations[0] else 1).to_numpy().astype(int)

    model.fit(
        X,
        {'label_classifier': y, 'domain_classifier': d},
        batch_size=128,
        epochs=200,
        shuffle=True
    )

    # Save the model
    model.save(
        f"../../models-{locations[0]}-{locations[1]}/website/grl_model.keras")
