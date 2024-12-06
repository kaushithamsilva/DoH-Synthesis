import random
import pandas as pd
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
