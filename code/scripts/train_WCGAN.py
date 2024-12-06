import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


@tf.keras.utils.register_keras_serializable()
class Generator(keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(Generator, self).__init__(**kwargs)
        # self.model = keras.Sequential([
        #     keras.layers.Dense(hidden_dim, input_shape=(input_dim,)),
        #     keras.layers.Reshape((1, hidden_dim)),
        #     keras.layers.GRU(output_dim, recurrent_dropout=0.1),

        # ])
        self.model = keras.Sequential([
            keras.layers.Dense(hidden_dim, input_shape=(input_dim,)),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(hidden_dim),
            keras.layers.LeakyReLU(0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Reshape((1, hidden_dim)),
            keras.layers.GRU(output_dim, recurrent_dropout=0.1),
        ])

    def call(self, inputs):
        return self.model(inputs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Critic(keras.Model):  # Renamed from Discriminator to Critic
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.model = keras.Sequential([
            keras.layers.Dense(hidden_dim, input_shape=(input_dim,)),
            keras.layers.RepeatVector(1),
            keras.layers.GRU(hidden_dim, recurrent_dropout=0.1),
            keras.layers.Dense(1)  # No sigmoid activation for WGAN
        ])

    def call(self, inputs):
        return self.model(inputs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def gradient_penalty(critic, real_samples, fake_samples, target_locations):
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)

    # Get random interpolation between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # Get critic output for interpolated inputs
        interpolated_input = tf.concat(
            [interpolated, target_locations], axis=1)
        critic_interpolated = critic(interpolated_input)

    # Calculate gradients with respect to inputs
    grads = gp_tape.gradient(critic_interpolated, interpolated)
    # Calculate gradient norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    gp = tf.reduce_mean(tf.square(norm - 1.0))

    return gp


@tf.function
def train_step(generator, critic, g_optimizer, c_optimizer, source_traces, source_locations,
               target_locations, target_traces, gp_weight=10.0, n_critic=5):

    for _ in range(n_critic):  # Train critic more times than generator
        noise = tf.random.normal(
            [tf.shape(source_traces)[0], tf.shape(source_traces)[1]])

        with tf.GradientTape() as tape:
            # Generate fake samples
            generator_inputs = tf.concat(
                [noise, source_traces, source_locations, target_locations], axis=1)
            synthetic_traces = generator(generator_inputs, training=True)

            # Get critic outputs
            real_input = tf.concat([target_traces, target_locations], axis=1)
            fake_input = tf.concat(
                [synthetic_traces, target_locations], axis=1)

            real_output = critic(real_input, training=True)
            fake_output = critic(fake_input, training=True)

            # Wasserstein loss
            critic_loss = tf.reduce_mean(
                fake_output) - tf.reduce_mean(real_output)

            # Gradient penalty
            gp = gradient_penalty(critic, target_traces,
                                  synthetic_traces, target_locations)
            critic_loss = critic_loss + gp_weight * gp

        # Apply critic gradients
        gradients = tape.gradient(critic_loss, critic.trainable_variables)
        c_optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

    # Train generator
    noise = tf.random.normal(
        [tf.shape(source_traces)[0], tf.shape(source_traces)[1]])

    with tf.GradientTape() as tape:
        generator_inputs = tf.concat(
            [noise, source_traces, source_locations, target_locations], axis=1)
        synthetic_traces = generator(generator_inputs, training=True)

        fake_input = tf.concat([synthetic_traces, target_locations], axis=1)
        fake_output = critic(fake_input, training=True)

        # Generator loss (negative of critic loss)
        generator_loss = -tf.reduce_mean(fake_output)

    # Apply generator gradients
    gradients = tape.gradient(generator_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

    return generator_loss, critic_loss


def prepare_training_data(df, locations):
    location_to_onehot = {location: tf.keras.utils.to_categorical(i, num_classes=len(locations))
                          for i, location in enumerate(locations)}

    train_data = {
        'source_trace': [],
        'source_location': [],
        'target_location': [],
        'target_trace': []
    }

    for source_location in locations:
        for target_location in locations:
            if source_location == target_location:
                continue

            source_data = df[df['Location'] == source_location].sort_values(by=[
                                                                            'Website'])
            target_data = df[df['Location'] == target_location].sort_values(by=[
                                                                            'Website'])

            for (_, source_row), (_, target_row) in zip(source_data.iterrows(), target_data.iterrows()):
                train_data['source_trace'].append(
                    source_row.iloc[2:].values.astype(np.float32))
                train_data['source_location'].append(
                    location_to_onehot[source_location].astype(np.float32))
                train_data['target_location'].append(
                    location_to_onehot[target_location].astype(np.float32))
                train_data['target_trace'].append(
                    target_row.iloc[2:].values.astype(np.float32))

    # Convert lists to numpy arrays
    for key in train_data:
        train_data[key] = np.array(train_data[key])

    return train_data


def train_wgan(df, locations, epochs, batch_size):
    train_data = prepare_training_data(df, locations)

    trace_dim = train_data['source_trace'].shape[1]
    location_dim = len(locations)

    generator_input_dim = 2 * trace_dim + 2 * location_dim
    generator = Generator(generator_input_dim, 32, trace_dim)
    generator.load_weights("../../models/GAN/WCGAN/generator-e400.keras")

    critic = Critic(trace_dim + location_dim, 32)
    critic.load_weights("../../models/GAN/WCGAN/critic-e400.keras")

    # Use RMSprop or Adam with lower learning rate for WGAN
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    dataset = tf.data.Dataset.from_tensor_slices((
        train_data['source_trace'],
        train_data['source_location'],
        train_data['target_location'],
        train_data['target_trace']
    )).shuffle(buffer_size=len(train_data['source_trace'])).batch(batch_size)

    for epoch in range(epochs):
        total_gen_loss = 0.0
        total_critic_loss = 0.0
        num_batches = 0

        for source_traces, source_locations, target_locations, target_traces in dataset:
            gen_loss, critic_loss = train_step(
                generator, critic, g_optimizer, c_optimizer,
                source_traces, source_locations, target_locations, target_traces
            )
            total_gen_loss += gen_loss
            total_critic_loss += critic_loss
            num_batches += 1

        avg_gen_loss = total_gen_loss / num_batches
        avg_critic_loss = total_critic_loss / num_batches

        print(
            f"Epoch [{epoch+1}/{epochs}], Critic Loss: {avg_critic_loss.numpy():.4f}, G Loss: {avg_gen_loss.numpy():.4f}")

    return generator, critic


if __name__ == '__main__':
    import init_gpu as init_gpu
    import init_dataset as init_dataset
    import pandas as pd

    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    generator, critic = train_wgan(
        train_df, locations, epochs=400, batch_size=128)

    generator.save("../../models/GAN/WCGAN/generator-e800.keras")
    critic.save("../../models/GAN/WCGAN/critic-e800.keras")
