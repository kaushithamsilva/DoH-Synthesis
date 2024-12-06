import tensorflow as tf
import os


def initialize_gpus():
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # List all physical GPUs available to TensorFlow
    gpus = tf.config.list_physical_devices('GPU')

    # Print the number of GPUs available
    print("Num GPUs Available: ", len(gpus))

    for gpu in gpus:
        # Enable memory growth for each GPU
        tf.config.experimental.set_memory_growth(gpu, True)

        # Enable async malloc for each GPU
        tf.config.experimental.set_synchronous_execution(False)
