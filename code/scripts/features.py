import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def extract_features_from_trace(trace):
    """
    Given a trace (a list of integers representing TLS record sizes),
    extract features as a dictionary containing counts for:
      - Unigrams: individual TLS record sizes.
      - Bigrams: pairs of consecutive TLS record sizes.
      - Burst features: grouping consecutive packets with the same sign.
    """
    features = {}

    # 1. Unigrams: count each TLS record size.
    for token in trace:
        key = f"U_{token}"
        features[key] = features.get(key, 0) + 1

    # 2. Bigrams: count each pair of consecutive TLS record sizes.
    for i in range(len(trace) - 1):
        key = f"B_{trace[i]}_{trace[i+1]}"
        features[key] = features.get(key, 0) + 1

    # 3. Burst features:
    #    Define a burst as a sequence of consecutive packets with the same sign.
    bursts = []
    if trace:
        current_burst = [trace[0]]
        for num in trace[1:]:
            # Check if the current number has the same sign as the last number in current burst.
            if (num >= 0 and current_burst[-1] >= 0) or (num < 0 and current_burst[-1] < 0):
                current_burst.append(num)
            else:
                bursts.append(current_burst)
                current_burst = [num]
        bursts.append(current_burst)

    # Burst Unigrams: use the sum of values in each burst.
    burst_unigrams = [sum(burst) for burst in bursts]
    for token in burst_unigrams:
        key = f"BU_{token}"
        features[key] = features.get(key, 0) + 1

    # Burst Bigrams: count consecutive burst sums as pairs.
    for i in range(len(burst_unigrams) - 1):
        key = f"BB_{burst_unigrams[i]}_{burst_unigrams[i+1]}"
        features[key] = features.get(key, 0) + 1

    return features


def extract_features_from_datasets(train_df, test_df):
    # Helper function to extract the trace from a DataFrame row.
    def get_trace_from_row(row):
        """
        Extracts the DNS trace from a DataFrame row.
        Assumes that the trace values are stored in the columns starting at index 2.
        Drops any NaN values and converts each entry to an integer.
        """
        trace = row.iloc[2:].dropna().tolist()
        return [int(x) for x in trace]

    # Process training dataset.
    train_feature_dicts = []
    for idx, row in train_df.iterrows():
        trace = get_trace_from_row(row)
        # Assumes extract_features_from_trace is defined.
        feats = extract_features_from_trace(trace)
        train_feature_dicts.append(feats)

    # Process test dataset.
    test_feature_dicts = []
    for idx, row in test_df.iterrows():
        trace = get_trace_from_row(row)
        # Assumes extract_features_from_trace is defined.
        feats = extract_features_from_trace(trace)
        test_feature_dicts.append(feats)

    # Use DictVectorizer: fit on training and transform both datasets.
    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(train_feature_dicts)
    X_test = vectorizer.transform(test_feature_dicts)

    print("Training feature matrix shape:", X_train.shape)
    print("Test feature matrix shape:", X_test.shape)
    print("Feature names:", vectorizer.get_feature_names_out())

    return X_train, X_test, vectorizer


if __name__ == "__main__":

    # Define the locations to use for the classification.
    locations = ['LOC2', 'LOC3']

    # load the dataset
    print("Loading Dataset...")
    df = pd.read_csv(
        f"../../dataset/processed/{locations[0]}-{locations[1]}-scaled-balanced.csv")

    length = len(df.columns) - 2  # subtract the two label columns
    # dataset for the classification
    source_location, target_location = locations

    target_df = df[df['Location'] == target_location]
    target_df.sort_values(by=['Website'], inplace=True)
    target_df.reset_index(drop=True, inplace=True)
    target_df.head(20)

    source_df = df[df['Location'] == source_location]
    source_df.sort_values(by=['Website'], inplace=True)
    source_df.reset_index(drop=True, inplace=True)

    # Extract features from the datasets.
    X_train, X_test, vectorizer = extract_features_from_datasets(
        source_df, target_df)

    # Train a Random Forest classifier and evaluate it.
    y_train = source_df['Website'].values
    y_test = target_df['Website'].values

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("Classification Report for Random Forest:\n",
          classification_report(y_test, y_pred_rf))
