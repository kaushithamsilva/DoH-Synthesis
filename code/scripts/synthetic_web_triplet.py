import numpy as np
import pandas as pd
import tensorflow as tf
import triplet_functions
import init_gpu
import init_dataset
from train_vae import ConvVAE_BatchNorm, Sampling


def make_synth_triplet_dataset(df, vae, batch_size=128):
    feats = df.iloc[:, 2:].to_numpy().astype(np.float32)
    webs = df['Website'].values
    locs = df['Location'].values
    idxs = np.arange(len(df), dtype=np.int32)

    # index pools
    by_web = {}
    by_loc_web = {}
    for i, (w, l) in enumerate(zip(webs, locs)):
        by_web.setdefault(w, []).append(i)
        by_loc_web.setdefault(l, {}).setdefault(w, []).append(i)
    for w in by_web:
        by_web[w] = np.array(by_web[w], dtype=np.int32)
    for l in by_loc_web:
        for w in by_loc_web[l]:
            by_loc_web[l][w] = np.array(by_loc_web[l][w], dtype=np.int32)

    def generator():
        while True:
            batch_a = np.random.choice(idxs, size=batch_size, replace=True)
            A, P, N = [], [], []
            for i in batch_a:
                w0 = webs[i]

                # positive: same class, any location but both from same location
                loc_choices = [l for l, wmap in by_loc_web.items(
                ) if w0 in wmap and len(wmap[w0]) >= 2]
                if loc_choices:
                    loc_p = np.random.choice(loc_choices)
                    pool_p = by_loc_web[loc_p][w0]
                    p1, p2 = np.random.choice(pool_p, size=2, replace=False)
                else:
                    p1, p2 = np.random.choice(
                        by_web[w0], size=2, replace=False)

                # negative: different class, any location but both from same location
                neg_classes = [w for w in by_web if w != w0]
                np.random.shuffle(neg_classes)
                for wn in neg_classes:
                    loc_n_choices = [l for l, wmap in by_loc_web.items(
                    ) if wn in wmap and len(wmap[wn]) >= 2]
                    if loc_n_choices:
                        loc_n = np.random.choice(loc_n_choices)
                        pool_n = by_loc_web[loc_n][wn]
                        n1, n2 = np.random.choice(
                            pool_n, size=2, replace=False)
                        break
                else:
                    pool_all_neg = np.concatenate(
                        [by_web[w] for w in neg_classes])
                    n1, n2 = np.random.choice(
                        pool_all_neg, size=2, replace=False)

                # encode and sample latent
                x_p = np.stack([feats[p1], feats[p2]], axis=0)
                _, _, z_p_batch = vae.encode(x_p)
                z1_p, z2_p = z_p_batch[0], z_p_batch[1]

                x_n = np.stack([feats[n1], feats[n2]], axis=0)
                _, _, z_n_batch = vae.encode(x_n)
                z1_n, z2_n = z_n_batch[0], z_n_batch[1]

                # interpolate
                eps_p = np.random.rand()
                z_p_interp = (z2_p - z1_p) * eps_p + z1_p
                eps_n = np.random.rand()
                z_n_interp = (z2_n - z1_n) * eps_n + z1_n

                # decode
                synth_p = vae.decode(z_p_interp[None, :])[0]
                synth_n = vae.decode(z_n_interp[None, :])[0]

                A.append(feats[i])
                P.append(synth_p)
                N.append(synth_n)

            yield ([np.stack(A, axis=0),
                    np.stack(P, axis=0),
                    np.stack(N, axis=0)],
                   np.stack(A, axis=0))

    D = feats.shape[1]  # feature dimension
    output_sig = (
        [  # a list of three inputs
            tf.TensorSpec((None, D), tf.float32),
            tf.TensorSpec((None, D), tf.float32),
            tf.TensorSpec((None, D), tf.float32),
        ],
        tf.TensorSpec((None, D), tf.float32)
    )
    # output signature for the dataset
    return tf.data.Dataset.from_generator(generator, output_signature=output_sig).prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    # 1) GPU init
    init_gpu.initialize_gpus()

    # 2) Load data
    locations = ['LOC2', 'LOC3']
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)

    input_dim = train_df.shape[1] - 2  # subtract the two label columns

    # load VAE
    vae = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/vae-e1000-mse1-kl0.0001-cl1.0-ldim96-hdim128.keras", custom_objects={
                                     'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    vae.trainable = False  # freeze VAE weights
    print("VAE loaded successfully!")
    print(vae.summary())

    # 4) Build triplet model
    base = triplet_functions.baseCNN(input_dim)
    model = triplet_functions.triplet_learning(base, input_dim)
    model.compile(optimizer='adam', loss=triplet_functions.triplet_loss_func)

    # 5) Create on-the-fly dataset
    batch_size = 128
    ds = make_synth_triplet_dataset(
        train_df, vae, batch_size=batch_size)
    steps = len(train_df) // batch_size
    print(f"Dataset created with {steps} steps per epoch.")

    # 6) Train
    print("Starting synthetic triplet model training...")
    model.fit(ds, steps_per_epoch=steps, epochs=1000)

    # 7) Save
    base.save(
        f"../../models/website/{locations[0]}-{locations[1]}-synthTriplet.keras")
    print("Synthetic triplet model training completed!")
