import numpy as np
import pandas as pd
import tensorflow as tf
import triplet_functions
import init_gpu
import init_dataset
from train_vae import ConvVAE_BatchNorm, Sampling


import numpy as np
import tensorflow as tf


def batched_encode(vae, x, batch_size=256):
    """
    Run vae.encode on x in smaller chunks to fit memory.
    Returns concatenated z_sample of shape (len(x), latent_dim).
    """
    z_list = []
    for i in range(0, len(x), batch_size):
        chunk = x[i:i+batch_size]
        _, _, z_chunk = vae.encode(chunk)
        z_list.append(z_chunk)
    return np.concatenate(z_list, axis=0)


def batched_decode(vae, z, batch_size=256):
    """
    Run vae.decode on z in smaller chunks.
    Returns concatenated reconstructions of shape (len(z), D).
    """
    x_list = []
    for i in range(0, len(z), batch_size):
        chunk = z[i:i+batch_size]
        x_chunk = vae.decode(chunk)
        x_list.append(x_chunk)
    return np.concatenate(x_list, axis=0)


def synth_triplets_offline(df, vae):
    """
    Precompute synthetic triplets for the entire df in a batched, vectorized way:
      - anchor: real sample
      - positive: VAE interpolation of two same-class samples from same location
      - negative: VAE interpolation of two different-class samples from same location

    Returns three NumPy arrays of shape (N, D): anchors, positives, negatives.
    """
    feats = df.iloc[:, 2:].to_numpy().astype(np.float32)  # (N, D)
    webs = df['Website'].values
    locs = df['Location'].values
    N, D = feats.shape

    # Build lookup pools
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

    # Pre-allocate index arrays
    p1 = np.zeros(N, dtype=np.int32)
    p2 = np.zeros(N, dtype=np.int32)
    n1 = np.zeros(N, dtype=np.int32)
    n2 = np.zeros(N, dtype=np.int32)

    # 1) Choose all p1,p2,n1,n2 indices
    for i in range(N):
        w0, l0 = webs[i], locs[i]

        # Positive: same class & same location if possible
        loc_choices = [l for l, wmap in by_loc_web.items(
        ) if w0 in wmap and len(wmap[w0]) >= 2]
        if loc_choices:
            lp = np.random.choice(loc_choices)
            pool = by_loc_web[lp][w0]
        else:
            pool = by_web[w0]
        p1[i], p2[i] = np.random.choice(pool, size=2, replace=False)

        # Negative: different class & same location if possible
        neg_classes = [w for w in by_web if w != w0]
        np.random.shuffle(neg_classes)
        found = False
        for wn in neg_classes:
            loc_n_choices = [l for l, wmap in by_loc_web.items()
                             if wn in wmap and len(wmap[wn]) >= 2]
            if loc_n_choices:
                ln = np.random.choice(loc_n_choices)
                pool_n = by_loc_web[ln][wn]
                n1[i], n2[i] = np.random.choice(pool_n, size=2, replace=False)
                found = True
                break
        if not found:
            pool_all = np.concatenate([by_web[w] for w in neg_classes])
            n1[i], n2[i] = np.random.choice(pool_all, size=2, replace=False)

    # 2) Stack into big batches for VAE encoding
    # positives: shape (2*N, D)
    batch_p = np.concatenate([
        feats[p1].reshape(N, 1, D),
        feats[p2].reshape(N, 1, D)
    ], axis=1).reshape(-1, D)
    # negatives: same
    batch_n = np.concatenate([
        feats[n1].reshape(N, 1, D),
        feats[n2].reshape(N, 1, D)
    ], axis=1).reshape(-1, D)

    encode_bs = 1024  # batch size for encoding
    decode_bs = 1024  # batch size for decoding
    # 3) One-shot VAE encoding
    # encode in smaller chunks
    zp_all = batched_encode(vae, batch_p, batch_size=encode_bs)
    zn_all = batched_encode(vae, batch_n, batch_size=encode_bs)

    zp_all = zp_all.reshape(N, 2, -1)
    zn_all = zn_all.reshape(N, 2, -1)

    latent_dim = zp_all.shape[-1]
    zp_all = zp_all.reshape(N, 2, latent_dim)
    zn_all = zn_all.reshape(N, 2, latent_dim)

    # 4) Vectorized interpolation
    eps_p = np.random.rand(N, 1)
    eps_n = np.random.rand(N, 1)
    zp_interp = zp_all[:, 0, :] + (zp_all[:, 1, :] - zp_all[:, 0, :]) * eps_p
    zn_interp = zn_all[:, 0, :] + (zn_all[:, 1, :] - zn_all[:, 0, :]) * eps_n

    # 5) One-shot VAE decoding
    synth_p = batched_decode(vae, zp_interp, batch_size=decode_bs)
    synth_n = batched_decode(vae, zn_interp, batch_size=decode_bs)

    # 6) Anchors are just the original features
    anchors = feats

    return anchors, synth_p, synth_n


if __name__ == '__main__':
    # init
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

    # load VAE
    vae = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/vae-e1000-mse1-kl0.0001-cl1.0-ldim96-hdim128.keras", custom_objects={
                                     'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    vae.trainable = False  # freeze VAE weights
    print("VAE loaded successfully!")
    print(vae.summary())

    # build triplet model
    base = triplet_functions.baseCNN(input_dim)

    # only for continuing training...
    custom_objects = {
        'ResidualBlock': triplet_functions.ResidualBlock
    }

    base = tf.keras.models.load_model(
        f"../../models-{locations[0]}-{locations[1]}/website/{locations[0]}-{locations[1]}-synth.keras", custom_objects=custom_objects)
    print("Base model loaded successfully!")

    model = triplet_functions.triplet_learning(base, input_dim)
    model.compile(optimizer='adam', loss=lambda y_true,
                  y_pred: triplet_functions.triplet_loss_func(y_true, y_pred, alpha=0.4))

    # training loop: regenerate every N epochs
    total_epochs = 50
    regenerate_every = 5

    for start in range(0, total_epochs, regenerate_every):
        end = min(start + regenerate_every, total_epochs)
        print(f"Generating synthetic triplets for epochs {start+1}-{end}...")
        A, P, N = synth_triplets_offline(train_df, vae)
        # fit for regenerate_every epochs
        model.fit(
            [A, P, N],    # inputs
            A,            # anchor as target
            epochs=start + regenerate_every,
            initial_epoch=start,
            batch_size=256,
            shuffle=True
        )

    # save
    base.save(
        f"../../models-{locations[0]}-{locations[1]}/website/{locations[0]}-{locations[1]}-synth.keras")
    print("Offline synthetic triplet training completed.")
