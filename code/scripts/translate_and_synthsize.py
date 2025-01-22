import tensorflow as tf
import pandas as pd
import numpy as np

@tf.keras.utils.register_keras_serializable()
class SWDLoss(tf.keras.losses.Loss):
    def __init__(self, latent_dim, num_projections=50):
        super(SWDLoss, self).__init__()
        self.latent_dim = latent_dim
        self.num_projections = num_projections

    def call(self, y_true, y_pred):
        # y_true: target embeddings (z_B), y_pred: predicted embeddings (z_A)
        projections = tf.random.normal([self.num_projections, self.latent_dim])
        projections = projections / tf.norm(projections, axis=1, keepdims=True)

        proj_A = tf.matmul(y_pred, projections, transpose_b=True)
        proj_B = tf.matmul(y_true, projections, transpose_b=True)

        proj_A_sorted = tf.sort(proj_A, axis=0)
        proj_B_sorted = tf.sort(proj_B, axis=0)

        return tf.reduce_mean(tf.square(proj_A_sorted - proj_B_sorted))
    
def get_z_embeddings(data, vae_model):
    embeddings = []
    chunk_size = 200
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        _, _, transformed_chunk = vae_model.encode(chunk)
        embeddings.append(transformed_chunk)

    return np.vstack(embeddings)

def get_z_translations(data, translator):
    translations = []
    chunk_size = 200
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        _, _, transformed_chunk = translator(chunk)
        translations.append(transformed_chunk)

    return np.vstack(translations) 

def generate_samples_from_a_z(z, vae_model, sampling, num_samples=100):
    decoded = vae_model.decode(z)
    z_mean, z_log_var, _ = vae_model.encode(decoded)

    # Repeat z_mean and z_log_var for num_samples
    z_mean = np.tile(z_mean, (num_samples, 1))
    z_log_var = np.tile(z_log_var, (num_samples, 1))

    # Generate samples in a single call
    z_samples = sampling((z_mean, z_log_var))

    return z_samples

if __name__ == '__main__':

    import init_gpu
    import init_dataset
    from train_vae import Sampling, ConvVAE_BatchNorm



    init_gpu.initialize_gpus()

    locations = ['LOC2', 'LOC3']
    source_location, target_location = locations

    print("Loading Dataset...")
    # load the dataset
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns

    # get train-test set
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    # load the pretrained models
    print("Loading Models...")
    latent_dim = 96
    vae_model = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/{locations[0]}-{locations[1]}-e800-mse1-kl0.01-cl1.0-ConvBatchNorm-ldim96-hdim128.keras", custom_objects={'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    translator = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/latent_translator/translator-kl0.01-e200.keras", custom_objects={'SWDLoss': SWDLoss})

    # use translator to translate the source latent embeddings from the vae. 
    # each website has the number of samples = eg: 99 for loc2-loc3 processed dataset
    print("Translating to target domain...")
    source_test_df = test_df[test_df['Location'] == source_location].sort_values(by=['Website'])
    source_z = get_z_embeddings(source_test_df.iloc[:, 2:])
    translated_z = get_z_translations(source_z)

    # for each translated z sample, synthesize 100 samples using the z_mean and z_var from the vae
    # Synthesize samples for each translated latent embedding
    synthesized_data = []
    print("synthesizing traces...")
    for idx, z in enumerate(translated_z):
        website = source_test_df.iloc[idx]['Website']
        print(website, idx)
        # Generate 100 samples from the current translated latent embedding
        samples = generate_samples_from_a_z(z[np.newaxis, :], vae_model, Sampling(), num_samples=100)

        # Flatten and structure the data
        for sample in samples:
            synthesized_data.append(
                [source_location, website] + sample.tolist()
            )

    # Convert the synthesized data into a DataFrame
    columns = ['Location', 'Website'] + [str(i) for i in range(128)]
    synthesized_df = pd.DataFrame(synthesized_data, columns=columns)

    # Save the synthesized data to a CSV file
    output_path = f"synthesized_samples_{source_location}_{target_location}.csv"
    synthesized_df.to_csv(output_path, index=False)

    print(f"Synthesized data saved to {output_path}") 
