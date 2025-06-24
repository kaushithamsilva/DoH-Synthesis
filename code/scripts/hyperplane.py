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


class Hyperplane:
    def __init__(self, model):
        self.model = model
        # Get the weights and biases from the model's dense layer
        W = self.model.layers[0].kernel  # Shape: (latent_dim, 1)
        b = self.model.layers[0].bias    # Shape: (1,)

        # The normal vector 'w' for a single-unit output is just the kernel
        # We need to squeeze it to remove the last dimension (latent_dim, 1) -> (latent_dim,)
        w = tf.squeeze(W, axis=-1)  # Normal vector: (latent_dim,)

        # The offset 'b' is simply the bias term
        b = tf.squeeze(b, axis=-1)  # Offset: scalar

        # Normalize w (this is good practice)
        w = w / tf.norm(w)

        self.w = w
        self.b = b

    def get_hyplerplane_params(self):
        return self.w, self.b

    def get_mirror_image(self, z):
        # Calculate the projection of z onto the hyperplane
        # The distance from a point z to the hyperplane w*x + b = 0 is (z*w + b) / ||w||
        # Since we normalized w, ||w|| = 1, so distance = z*w + b

        # Ensure z and w have compatible shapes for dot product or matrix multiplication
        # z is typically (batch_size, latent_dim)
        # w is (latent_dim,)

        # We need tf.einsum or tf.reduce_sum(z * self.w, axis=-1) for dot product over batch
        # Or simply tf.matmul(z, tf.expand_dims(self.w, axis=-1)) for (batch_size, 1) result

        # The current implementation of z_dist expects z to be (latent_dim,) for np.dot
        # But in extrapolation, z will be (1, latent_dim)

        # Let's adjust z_dist calculation to be robust to batch dimension (1, latent_dim)
        # Assuming z is (batch_size, latent_dim)
        z_dist = tf.reduce_sum(z * self.w, axis=-1) + self.b
        # z_dist shape will be (batch_size,)

        # For mirror image, we need to apply this distance to each element of the batch.
        # Reshape z_dist to (batch_size, 1) so it broadcasts correctly with self.w (latent_dim,)
        z_dist_expanded = tf.expand_dims(
            z_dist, axis=-1)  # Shape: (batch_size, 1)

        z_mirror = z - 2 * z_dist_expanded * self.w
        return z_mirror
