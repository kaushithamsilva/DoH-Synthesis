import tensorflow as tf
import pandas as pd
import numpy as np

@tf.keras.utils.register_keras_serializable()
class SWDLoss(tf.keras.losses.Loss):
    def __init__(self, latent_dim, num_projections=50, **kwargs):
        super(SWDLoss, self).__init__(**kwargs)
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
    
    
    def get_config(self):
        config = super(SWDLoss, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "num_projections": self.num_projections
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract latent_dim and pass it to the constructor
        # TODO: I've hard coded this latent dim for now, there is an issue when loading the translator model. 
        latent_dim = config.pop('latent_dim', 96)
        num_projections = config.pop('num_projections', 50)
        return cls(latent_dim=latent_dim, num_projections=num_projections, **config)
    
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
        translated_chunk = translator(chunk)
        translations.append(translated_chunk)

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

# Function to synthesize and write data in chunks
def synthesize_and_write_to_csv(translated_z, source_test_df, vae_model, sampling, output_path, target_location, latent_dim, chunk_size=5, num_samples=1000):
    print("Synthesizing traces and writing to CSV...")
    
    # Write CSV header
    columns = ['Location', 'Website'] + [str(i) for i in range(latent_dim)]
    with open(output_path, mode='w') as file:
        file.write(','.join(columns) + '\n')  # Write the header line

    for idx in range(0, len(translated_z), chunk_size):
        chunk = translated_z[idx:idx + chunk_size]
        websites = source_test_df.iloc[idx:idx + chunk_size]['Website'].values
        
        synthesized_data = []
        for i, z in enumerate(chunk):
            website = websites[i]
            samples = generate_samples_from_a_z(z[np.newaxis, :], vae_model, sampling, num_samples=num_samples)
            for sample in samples:
                synthesized_data.append(
                    [target_location, website] + sample.numpy().tolist()
                )
        
        # Append chunk to CSV file
        chunk_df = pd.DataFrame(synthesized_data, columns=columns)
        chunk_df.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Appended {len(synthesized_data)} entries to {output_path} (Chunk {idx // chunk_size + 1})")

    print(f"Synthesized data saved to {output_path}")

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
    swd_loss = SWDLoss(latent_dim)
    vae_model = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/{locations[0]}-{locations[1]}-e800-mse1-kl0.01-cl1.0-ConvBatchNorm-ldim96-hdim128.keras", custom_objects={'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    translator = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/latent_translator/translator-kl0.01-e200.keras", custom_objects={'SWDLoss': swd_loss})

    # use translator to translate the source latent embeddings from the vae. 
    # each website has the number of samples = eg: 99 for loc2-loc3 processed dataset
    print("Translating to target domain...")
    source_test_df = test_df[test_df['Location'] == source_location].sort_values(by=['Website'])
    source_z = get_z_embeddings(source_test_df.iloc[:, 2:], vae_model)
    translated_z = get_z_translations(source_z, translator)

    # for each translated z sample, synthesize 100 samples using the z_mean and z_var from the vae
    # Synthesize samples for each translated latent embedding
    output_path = f"../../synthesized/{target_location}-VAE-Sampling-Z-Translations.csv"

    # Call the function to synthesize and write data
    synthesize_and_write_to_csv(
        translated_z=translated_z,
        source_test_df=source_test_df,
        vae_model=vae_model,
        sampling=Sampling(),
        output_path=output_path,
        target_location=target_location,
        latent_dim=latent_dim,
        chunk_size=5,  # Adjust chunk size as needed
        num_samples=10
    ) 
