# Downstream classification tasks
# Classification: Trained on synthetic data, test on real data
import pandas as pd
import numpy as np
import os
import joblib
import classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
import xgboost

if __name__ == '__main__':
    import init_dataset
    print("Loading Dataset...")
    # load the dataset
    locations = ['LOC2', 'LOC3']
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns
    train_df, test_df, train_web_samples, test_web_samples = init_dataset.get_sample(df, locations, range(1500), 1200)
    source_location, target_location = locations

    # data preprocessing for source, real target, and synthetic data
    target_df = test_df[test_df['Location'] == target_location]
    target_df.sort_values(by=['Website'], inplace=True)
    target_df.reset_index(drop=True, inplace=True)
    target_df.head(20)

    source_df = test_df[test_df['Location'] == source_location]
    source_df.sort_values(by=['Website'], inplace=True)
    source_df.reset_index(drop=True, inplace=True)


    synthetic_df = pd.read_csv(f'../../synthesized/{target_location}-VAE-Sampling.csv')
    synthetic_df['Location'] = target_location # TODO: this is just a correction of an error done when synthesizing. 
    synthetic_df = synthetic_df[synthetic_df['Location'] == target_location]
    synthetic_df.sort_values(by=['Website'], inplace=True)
    synthetic_df.reset_index(drop=True, inplace=True)


    le = LabelEncoder()

    # synthetic data
    X_train = synthetic_df.iloc[:, 2:]
    y_train = le.fit_transform(synthetic_df.Website)
    X_test = target_df.iloc[:, 2:]
    y_test = le.transform(target_df.Website)

    # Train and evaluate the model
    model = xgboost.XGBClassifier(max_depth=10, n_estimators=30)
    print("Trained completely on the synthesized data: ")
    accuracy, precision, recall, f1_score, cm = classification.evaluate_classification_model(X_train, y_train, X_test, y_test, model)

    save_path = f"../../models-{source_location}-{target_location}/classification/website/"
    model_name = 'xgb-synthetic-only-d10-n30'

    # Ensure the save path directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the model
    model_path = os.path.join(save_path, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
