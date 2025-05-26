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
    # 1) Input is 2D (batch, input_dim)
    inputs = keras.Input(shape=(input_dim,), name="feature_input")

    # 2) Expand to 3D for Conv1D: (batch, input_dim, 1)
    x = layers.Reshape((input_dim, 1))(inputs)

    # 3) Now pass through your baseCNN
    features = triplet_functions.baseCNN(input_dim)(x)

    # 4) Label classifier head
    label_preds = layers.Dense(
        num_classes, activation='softmax', name='label_classifier'
    )(features)

    # 5) Domain classifier via GRL
    x_grl = GradientReversalLayer(lambda_=grl_lambda)(features)
    x = layers.Dense(64, activation='relu')(x_grl)
    domain_preds = layers.Dense(
        num_locations, activation='softmax', name='domain_classifier'
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=[label_preds, domain_preds],
        name='feature_extractor_grl'
    )


if __name__ == '__main__':
    import init_gpu
    import init_dataset
    import pandas as pd
    import numpy as np

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

    assert set(np.unique(d)).issubset({0, 1})
    assert not np.any(np.isnan(d))

    # Example usage
    input_dim = 128  # e.g., feature vector length
    num_classes = 1500
    num_locations = 2

    grl_lambda = 0.1

    model = build_grl_model(input_dim, num_classes, num_locations, grl_lambda)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss={
            'label_classifier': 'sparse_categorical_crossentropy',
            'domain_classifier': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'label_classifier': 1.0,
            'domain_classifier': 0.1
        },
        metrics={
            'label_classifier': 'accuracy',
            'domain_classifier': 'accuracy'
        }
    )

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
