import numpy as np
import pandas as pd
import tensorflow as tf
import triplet_functions
import init_gpu
import init_dataset
from train_vae import ConvVAE_BatchNorm, Sampling


def synth_triplets_offline(df, vae, num_triplets=4):
    """
    Precompute synthetic triplets for the entire df:
    - anchor: real sample
    - positive: VAE interpolation of two same-class samples from same location
    - negative: VAE interpolation of two different-class samples from same location
    Returns: anchors, positives, negatives (numpy arrays)
    """
    feats = df.iloc[:, 2:].to_numpy().astype(np.float32)
    webs = df['Website'].values
    locs = df['Location'].values
    N = len(df)

    # build pools
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

    anchors = []
    positives = []
    negatives = []

    for i in range(N):
        w0, l0 = webs[i], locs[i]

        # positive selection
        loc_choices = [l for l, wmap in by_loc_web.items(
        ) if w0 in wmap and len(wmap[w0]) >= 2]
        if loc_choices:
            lp = np.random.choice(loc_choices)
            p1, p2 = np.random.choice(
                by_loc_web[lp][w0], size=2, replace=False)
        else:
            p1, p2 = np.random.choice(by_web[w0], size=2, replace=False)

        # negative selection
        neg_classes = [w for w in by_web if w != w0]
        np.random.shuffle(neg_classes)
        for wn in neg_classes:
            loc_n_choices = [l for l, wmap in by_loc_web.items(
            ) if wn in wmap and len(wmap[wn]) >= 2]
            if loc_n_choices:
                ln = np.random.choice(loc_n_choices)
                n1, n2 = np.random.choice(
                    by_loc_web[ln][wn], size=2, replace=False)
                break
        else:
            pool_all_neg = np.concatenate([by_web[w] for w in neg_classes])
            n1, n2 = np.random.choice(pool_all_neg, size=2, replace=False)

        # synthesize via VAE
        xp = np.stack([feats[p1], feats[p2]], axis=0)
        _, _, zp = vae.encode(xp)
        z1p, z2p = zp[0], zp[1]
        xn = np.stack([feats[n1], feats[n2]], axis=0)
        _, _, zn = vae.encode(xn)
        z1n, z2n = zn[0], zn[1]

        ep = np.random.rand()
        ezp = (z2p - z1p)*ep + z1p
        en = np.random.rand()
        ezn = (z2n - z1n)*en + z1n

        synth_p = vae.decode(ezp[None, :])[0]
        synth_n = vae.decode(ezn[None, :])[0]

        anchors.append(feats[i])
        positives.append(synth_p)
        negatives.append(synth_n)

    return (np.stack(anchors),
            np.stack(positives),
            np.stack(negatives))


if __name__ == '__main__':
    # init
    init_gpu.initialize_gpus()
    locations = ['LOC2', 'LOC3']
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")
    train_df, test_df, _, _ = init_dataset.get_sample(
        df, locations, range(1500), 1200)
    input_dim = train_df.shape[1] - 2

    # load VAE
    vae = tf.keras.models.load_model(f"../../models-{locations[0]}-{locations[1]}/vae/ci_vae/ConvBased/domain_and_class/vae-e1000-mse1-kl0.0001-cl1.0-ldim96-hdim128.keras", custom_objects={
                                     'ConvVAE_BatchNorm': ConvVAE_BatchNorm, 'Sampling': Sampling})
    vae.trainable = False  # freeze VAE weights
    print("VAE loaded successfully!")
    print(vae.summary())

    # build triplet model
    base = triplet_functions.baseCNN(input_dim)
    model = triplet_functions.triplet_learning(base, input_dim)
    model.compile(optimizer='adam', loss=triplet_functions.triplet_loss_func)

    # training loop: regenerate every N epochs
    total_epochs = 1000
    regenerate_every = 25

    for start in range(0, total_epochs, regenerate_every):
        end = min(start + regenerate_every, total_epochs)
        print(f"Generating synthetic triplets for epochs {start+1}-{end}...")
        A, P, N = synth_triplets_offline(train_df, vae, num_triplets=1)
        # fit for regenerate_every epochs
        model.fit(
            [A, P, N],    # inputs
            A,            # anchor as target
            epochs=regenerate_every,
            initial_epoch=start,
            batch_size=128,
            shuffle=True
        )

    # save
    base.save(
        f"../../models/website/{locations[0]}-{locations[1]}-synth.keras")
    print("Offline synthetic triplet training completed.")
