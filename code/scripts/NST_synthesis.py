import tensorflow as tf
from train_vae import filter_and_sort_data, VAE, Sampling
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class TrafficSynthesisModel(tf.keras.Model):
    def __init__(self, vae_model: VAE, website_model: tf.keras.Model, location_classifier: tf.keras.Model, latent_location_classifier: tf.keras.Model, v_source: np.array, target_location: str):
        super(TrafficSynthesisModel, self).__init__()
        self.website_model = website_model
        self.location_classifier = location_classifier
        self.latent_location_classifier = latent_location_classifier
        self.website_embedding = website_model(v_source)
        self.v_synth = v_source
        self.epsilon = 1e-6  # A small value to avoid division by zero
        self.website_losses = []
        self.location_losses = []
        self.original_losses = []

        if target_location == 'LOC1':
            self.target_location = tf.constant(0.0, dtype=tf.float32)
        elif target_location == 'LOC2':
            self.target_location = tf.constant(1.0, dtype=tf.float32)
        else:
            raise Exception('Invalid Location')

        self.vae_model = vae_model

        _, _, latent_embedding = vae_model.encode(v_source)
        self.latent_embedding = tf.Variable(latent_embedding)

        # fix the weights of the other models
        self.website_model.trainable = False
        self.location_classifier.trainable = False
        self.vae_model.trainable = False

        #  # Annealing parameters
        self.max_annealing_epoch = 200
        self.max_reg_weight = 0.01

        # website triplet weight
        self.max_website_weight = 0.5

    def get_regularization_weight(self, epoch):
        # Linearly increase the regularization weight until max_annealing_epoch
        if epoch < self.max_annealing_epoch:
            return self.max_reg_weight * (epoch / self.max_annealing_epoch)
        else:
            return self.max_reg_weight

    def get_website_weight(self, epoch):
        # Linearly increase the regularization weight until max_annealing_epoch
        if epoch < self.max_annealing_epoch:
            return self.max_website_weight * (epoch / self.max_annealing_epoch)
        else:
            return self.max_website_weight

    def get_euclidean_distance(self, vector_a, vector_b):
        # return tf.reduce_sum(tf.square(vector_a - vector_b))
        return tf.reduce_mean(tf.square(vector_a - vector_b))

    def total_loss(self, epoch):
        # print("Decoding Latent Embedding...")
        self.v_synth = self.vae_model.decode(self.latent_embedding)

        # print("Website Euclidean Loss...")
        website_loss = self.get_euclidean_distance(
            self.website_embedding, self.website_model(self.v_synth))

        # synthesized trace location classification error
        predicted_location = tf.clip_by_value(self.location_classifier(self.v_synth)[
                                              0][0], self.epsilon, 1 - self.epsilon)
        classification_loss = - (self.target_location * tf.math.log(predicted_location) + (
            1 - self.target_location) * tf.math.log(1 - predicted_location))

        predicted_location = tf.clip_by_value(self.latent_location_classifier(
            self.latent_embedding)[0][0], self.epsilon, 1 - self.epsilon)
        latent_classification_loss = - (self.target_location * tf.math.log(
            predicted_location) + (1 - self.target_location) * tf.math.log(1 - predicted_location))

        self.website_losses.append(website_loss)
        self.location_losses.append(classification_loss)

        # l2 regularization
        l2_loss = tf.reduce_sum(tf.square(self.latent_embedding))

        # print(f"Web Loss: {website_loss}, Cl Loss : {classification_loss}, Latent_cl_loss: {latent_classification_loss}")
        total_loss = latent_classification_loss + website_loss + 0.001 * l2_loss
        return total_loss, website_loss, classification_loss, latent_classification_loss, l2_loss

    def train_step(self, epoch):
        with tf.GradientTape() as tape:
            total_loss, website_loss, classification_loss, latent_classification_loss, l2_loss = self.total_loss(
                epoch)
        gradients = tape.gradient(total_loss, [self.latent_embedding])
        self.optimizer.apply_gradients(zip(gradients, [self.latent_embedding]))
        return total_loss, website_loss, classification_loss, latent_classification_loss, l2_loss

    def fit(self, target_sample_data: np.array, epochs=1000, lr=0.01):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        _, _, z = self.vae_model.encode(target_sample_data)
        for epoch in range(epochs):

            loss, web_loss, location_trace_classification_loss, latent_classification_loss, l2_loss = self.train_step(
                epoch)

            # Early stopping if loss is oscillating
            if (latent_classification_loss <= 1e-2) or (l2_loss <= 4):
                print(
                    f"Early stopping triggered at epoch {epoch}. Loss: {loss.numpy()}")
                print(f"\tWeb Loss: {web_loss:.4f}, Synth Trace Classification Loss: {location_trace_classification_loss:.4f}, Latent Location Classification: {latent_classification_loss:.4f}, L2: {l2_loss:.4f}")
                print(
                    f"\tLatent Space Distance: {self.get_euclidean_distance(z, self.latent_embedding)}")
                break

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch}, Loss: {loss.numpy()} Loss with Original: {self.get_euclidean_distance(self.v_synth, target_sample_data)}")
                print(f"\tWeb Loss: {web_loss:.4f}, Synth Trace Classification Loss: {location_trace_classification_loss:.4f}, Latent Location Classification: {latent_classification_loss:.4f}, L2: {l2_loss:.4f}")
                print(
                    f"\tLatent Space Distance: {self.get_euclidean_distance(z, self.latent_embedding)}")
        return self.v_synth.numpy(), location_trace_classification_loss, latent_classification_loss


if __name__ == '__main__':
    import init_gpu as init_gpu
    import init_dataset as init_dataset
    init_gpu.initialize_gpus()

    locations = ['LOC1', 'LOC2']

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # load models
    vae_model = tf.keras.models.load_model(
        "../../models/vae/LOC1-LOC2-e400-mse1-kl0.01.keras", custom_objects={'VAE': VAE, 'Sampling': Sampling})
    web_model = tf.keras.models.load_model(
        f"../../models/website/{locations[0]}-{locations[1]}-baseGRU-epochs300-train_samples1200-triplet_samples5.keras")
    location_classifier = tf.keras.models.load_model(
        f"../../models/classification/location/dense.keras")
    latent_location_classifier = tf.keras.models.load_model(
        f"../../models/classification/location/latent_classifier.keras")

    random_seed = 42
    target_location = 'LOC2'
    source_location = 'LOC1'
    synthesized_vectors = []
    num_samples = 500

    i = 0
    while i != num_samples:
        print(f"{i} of {num_samples}...")
        target_website = 513

        target_data = test_df[(test_df['Website'] == target_website) & (
            test_df['Location'] == target_location)].iloc[:, 2:].sample(random_state=random_seed)

        target_data = target_data.to_numpy().reshape(
            1, length, 1)  # Reshape to (1, length, 1)

        source_trace = test_df[(test_df['Location'] == source_location) & (
            test_df['Website'] == target_website)].iloc[:, 2:].sample()

        source_trace = source_trace.to_numpy().reshape(1, length, 1)

        # initialize model
        print("Initializing TSM Model...")
        tsm_model = TrafficSynthesisModel(
            vae_model, web_model, location_classifier, latent_location_classifier, source_trace, target_location)
        # synthesize
        epochs = 500
        lr = 0.1

        print("Synthesizing Trace...")
        synthesized_vector, location_trace_classification_loss, latent_classification_loss = tsm_model.fit(
            target_data, epochs=epochs, lr=lr)

        # if not a good synthesis
        if (latent_classification_loss > 0.2) or (location_trace_classification_loss > 0.2):
            print("Neglecting this one, not a good synthesis...")
            continue

        synthesized_vectors.append(synthesized_vector)

    synthesized_vectors = np.array(
        synthesized_vectors).reshape(num_samples, length)
    np.save("synth_nst.npy", synthesized_vectors)
