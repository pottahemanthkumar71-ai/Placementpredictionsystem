from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data
)
from sklearn.ensemble import RandomForestClassifier


def create_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        oob_score=True
    )
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)

    print("\nDecision Tree trained successfully!")

    return model


def evaluate_model(model, x_test, y_test):
    accuracy = model.score(x_test, y_test)

    print("Test Accuracy: ", accuracy)
    print("OOB Score: ", model.oob_score_)
    return accuracy


def main():
    df = load_data()
    print("\nOriginal Dataset Shape:")
    print(df.shape)

    x_train, x_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nTraining Data Shape:")
    print(x_train.shape)

    print("\nTesting Data Shape:")
    print(x_test.shape)

    # Identify Features
    numerical_features, categorical_features = (
        identify_features(x_train)
    )
    print("\nNumerical Features:")

    for feature in numerical_features:
        print(" -", feature)

    print("\nNumber of Numerical Features:")
    print(len(numerical_features))

    print("\nCategorical Features:")

    for feature in categorical_features:
        print(" -", feature)

    print("\nNumber of Categorical Features:")
    print(len(categorical_features))

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    x_train, x_test, imputer = handle_missing_values(
        x_train,
        x_test,
        numerical_features
    )
    print("\nMissing Values Handling Completed.")

    x_train, x_test, one_hot_encoded = one_hot_encode_data(
        x_train,
        x_test,
        one_hot_features
    )
    print("\nOne-Hot Encoding Completed.")

    x_train, x_test, ordinal_encoded = ordinal_encode_data(
        x_train,
        x_test,
        ordinal_features
    )
    print("\nOrdinal Encoding Completed.")

    model = create_model()

    model = train_model(model, x_train, y_train)

    evaluate_model(model, x_test, y_test)