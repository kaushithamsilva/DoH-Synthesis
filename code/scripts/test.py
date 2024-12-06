import tensorflow as tf


def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Number of GPUs available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
    else:
        print("No GPU available.")


check_gpus()
