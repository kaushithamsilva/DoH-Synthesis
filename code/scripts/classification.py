from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification_model(X_train, y_train, X_test, y_test, model) -> tuple[float, float, float, np.array]:
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    precision = metrics.precision_score(
        y_test, y_pred, average='macro', zero_division=0)
    recall = metrics.recall_score(
        y_test, y_pred, average='macro', zero_division=0)
    f1_score = metrics.f1_score(
        y_test, y_pred, average='macro', zero_division=0
    )
    print(f"Accuracy: {accuracy * 100.0:.2f}, F1 Score: {f1_score * 100.0: .2f}, Precision: {precision * 100.0: .2f}, Recall: {recall * 100.0: .2f}")
    return accuracy, precision, recall, f1_score, confusion_matrix


def preprocess_data_for_platform_classification(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()

    X_train = train_df.iloc[:, 2:]
    X_test = test_df.iloc[:, 2:]

    y_train = le.fit_transform(train_df['Location'])
    y_test = le.transform(test_df['Location'])

    return (X_train, X_test, y_train, y_test, le)


def evaluate_classification_neural_model(X_test, y_test, model):
    # Generate predictions on the test set
    y_pred_prob = model.predict(X_test)  # Predicted probabilities
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100.0:.2f}, F1: {f1 * 100:.2f} Precision: {precision * 100.0:.2f}, Recall: {recall * 100.0:.2f}")
    return accuracy, precision, recall, f1, conf_matrix


def preprocess_data_for_web_classification(df, train_location, test_location):
    le = LabelEncoder()
    X_train = df[df['Location'] == train_location].drop(
        ['Location', 'Website'], axis=1)
    X_test = df[df['Location'] == test_location].drop(
        ['Location', 'Website'], axis=1)
    y_train = df[df['Location'] == train_location]['Website']
    y_test = df[df['Location'] == test_location]['Website']

    y_test = le.fit_transform(y_test)
    y_train = le.fit_transform(y_train)

    return X_train, y_train, X_test, y_test, le


def show_confusion_matrix_heatmap(confusion_matrix, label_encoder, title):
    f, ax = plt.subplots(constrained_layout=True)
    sns.heatmap(confusion_matrix, annot=True, linewidths=0.5, linecolor="red", fmt=".0f",
                ax=ax, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.title(f"Confusion Matrix - {title}")
    plt.show()
