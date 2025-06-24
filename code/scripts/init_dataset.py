import random
import pandas as pd
from typing import Tuple, List
import numpy as np
import tensorflow as tf
random_seed = 42


def get_sample(df: pd.DataFrame, train_locations: list[str], all_websites: list[int] = range(1500), num_websites: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    random.seed(random_seed)
    train_web_samples = random.sample(all_websites, num_websites)
    test_web_samples = list(set(all_websites) - set(train_web_samples))

    print(f"Training Websites: {train_web_samples}")
    print(f"Training Locations: {train_locations}")

    train_df = df[df["Location"].isin(
        train_locations) & df["Website"].isin(train_web_samples)]
    train_df.sort_values(by=["Location"], inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df = df[df["Location"].isin(train_locations) & (df["Website"].isin(
        test_web_samples))]

    return train_df, test_df, train_web_samples, test_web_samples


def get_seen_unseen_df(train_df: pd.DataFrame, test_df: pd.DataFrame, source_location='LOC1', target_location='LOC2'):
    seen_test_df = test_df[test_df['Location'] == source_location]
    seen_df = pd.concat((seen_test_df, train_df))
    unseen_test_df = test_df[test_df['Location'] == target_location]
    return seen_df, unseen_test_df


def get_train_test_dataset(
    csv_path: str,
    num_train: int = 1200,
    num_test: int = 300,
    batch_size: int = 128,
    random_seed: int = 42
) -> Tuple[
    tf.data.Dataset,    # train dataset
    tf.data.Dataset,    # test dataset
    List[int],          # train website IDs
    List[int],          # test website IDs
    List[str]           # attribute names
]:
    """
    1) Loads your preprocessed CSV with columns [Website, Location, Resolver, Client, Platform, 0..127]
    2) Splits by unique Website IDs into train/test (num_train / num_test).
    3) Builds binary targets for each unique metadata value.
    4) Wraps into tf.data.Datasets, shuffles & batches the train set.

    Returns:
      train_ds, test_ds, train_websites, test_websites, attribute_names
    """
    # --- 1) Load CSV ---
    df = pd.read_csv(csv_path, index_col=0)

    # --- 2) Split by Website ID ---
    all_websites = df["Website"].unique().tolist()
    random.seed(random_seed)
    sampled = random.sample(all_websites, num_train + num_test)
    train_websites = sampled[:num_train]
    test_websites = sampled[num_train:]

    train_df = df[df["Website"].isin(train_websites)].reset_index(drop=True)
    test_df = df[df["Website"].isin(test_websites)].reset_index(drop=True)

    print(f"→ {len(train_websites)} train sites, {len(test_websites)} test sites")

    # --- 3) Extract features and build binary targets ---
    feature_cols = [str(i) for i in range(128)]
    meta_cols = ["Location", "Resolver", "Client", "Platform"]

    X_train = train_df[feature_cols].astype("float32").values
    X_test = test_df[feature_cols].astype("float32").values

    # Build one binary column per unique metadata value
    attribute_names = []
    y_parts_train = []
    y_parts_test = []

    for col in meta_cols:
        uniques = df[col].unique()
        for val in uniques:
            name = f"{col.lower()}_{val.lower().replace(' ', '_')}"
            attribute_names.append(name)
            y_parts_train.append(
                (train_df[col] == val).astype("float32").values)
            y_parts_test .append(
                (test_df[col] == val).astype("float32").values)

    # Stack into (n_samples, n_attributes)
    y_train = np.stack(y_parts_train, axis=1)
    y_test = np.stack(y_parts_test,  axis=1)

    # --- 4) Build tf.data.Datasets ---
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(
        buffer_size=1024, seed=random_seed).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds, train_websites, test_websites, attribute_names
