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


# Function to randomly sample pairs from matching websites
def sample_random_pairs(source_embeddings, target_embeddings, source_labels, target_labels, samples_per_website):
    unique_websites = np.unique(source_labels)
    sampled_source = []
    sampled_target = []

    for website in unique_websites:
        # Get indices for the current website in both source and target
        source_indices = np.where(source_labels == website)[0]
        target_indices = np.where(target_labels == website)[0]
        
        # Randomly sample indices
        source_sampled = np.random.choice(source_indices, size=samples_per_website, replace=True)
        target_sampled = np.random.choice(target_indices, size=samples_per_website, replace=True)
        
        # Collect the corresponding embeddings
        sampled_source.append(source_embeddings[source_sampled])
        sampled_target.append(target_embeddings[target_sampled])

    # Concatenate all sampled data
    sampled_source = np.vstack(sampled_source)
    sampled_target = np.vstack(sampled_target)

    return sampled_source, sampled_target
    
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

    # load the pretraine vae model
    print("Loading VAE...")
    latent_dim = 96
    vae_model = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/{locations[0]}-{locations[1]}-e800-mse1-kl0.01-cl1.0-ConvBatchNorm-ldim96-hdim128.keras", custom_objects={'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    
    
    print("Preprocessing the trace pairs...")
    # Group embeddings by 'Website'
    source_location, target_location = locations
    source_data = train_df[(train_df['Location'] == source_location)].iloc[:, 2:]
    target_data = train_df[(train_df['Location'] == target_location)].iloc[:, 2:]

    source_labels = train_df[(train_df['Location'] == source_location)]['Website'].values
    target_labels = train_df[(train_df['Location'] == target_location)]['Website'].values

    # Generate source and target latent embeddings
    source_latent_embeddings = get_z_embeddings(source_data, vae_model)
    target_latent_embeddings = get_z_embeddings(target_data, vae_model)

    # Define the translator model
    translator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim, )),
        tf.keras.layers.Dense(latent_dim * 2, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(latent_dim * 4, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(latent_dim, activation=None)  # No activation for residual connection
    ])

    # Compile the translator model
    translator.compile(optimizer='adam', loss='mse')

    # Training loop with dynamic sampling
    epochs = 200
    samples_per_website = 200

    # Instantiate the SWD loss
    swd_loss = SWDLoss(latent_dim=latent_dim, num_projections=32)

    # Compile the translator model with the SWD loss
    translator.compile(optimizer='adam', loss=swd_loss)

    for epoch in range(epochs):

        print(f"{epoch + 1} of {epochs}")
        # Randomly sample pairs for the current epoch
        sampled_source, sampled_target = sample_random_pairs(
            source_latent_embeddings, 
            target_latent_embeddings, 
            source_labels, 
            target_labels, 
            samples_per_website
        )
        
        # Train on the sampled data
        translator.fit(sampled_source, sampled_target, epochs=1, verbose=1, shuffle=True, batch_size=32)

    print("Saving model..")
    translator.save(f"../../models-{locations[0]}-{locations[1]}/latent_translator/translator-kl0.01-e200.keras")