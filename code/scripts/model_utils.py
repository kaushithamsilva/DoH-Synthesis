import os
import tensorflow as tf

def save_model(model, file_path, file_name):
    # Create the full file path by joining the directory path with the file name
    full_file_path = os.path.join(file_path, file_name)

    # Extract the directory from the full file path
    directory = os.path.dirname(full_file_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the model
    model.save(f"{full_file_path}.keras")
    print(f"Model saved to {full_file_path}")

