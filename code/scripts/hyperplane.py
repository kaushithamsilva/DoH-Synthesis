import tensorflow as tf


def get_hyperplane(domain_discriminator):
    # Get the weights and biases from the domain_discriminator's dense layer
    # Shape: (latent_dim, 2) in TensorFlow
    W = domain_discriminator.layers[0].kernel
    b = domain_discriminator.layers[0].bias   # Shape: (2,)

    # Calculate the hyperplane parameters.  Note the transpose and slicing.
    w = W[:, 0] - W[:, 1]   # Normal vector: (latent_dim,)
    b = b[0] - b[1]       # Offset: scalar

    # Normalize w (optional, but often helpful)
    w = w / tf.norm(w)
    return w, b
