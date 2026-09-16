from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.data.load_data import load_data
import pandas as pd
import os


def split_data(df):
    X = df.drop(columns=["PlacementStatus"])
    y = df["PlacementStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def handle_missing_values(X_train, X_test, numerical_features):
    imputer = SimpleImputer(strategy="median")

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    X_train[numerical_features] = imputer.fit_transform(
        X_train[numerical_features]
    )

    # Transform test data using the same imputer
    X_test[numerical_features] = imputer.transform(
        X_test[numerical_features]
    )

    return X_train, X_test, imputer


def identify_features(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return numerical_features, categorical_features


def standardize_data(X_train, X_test, numerical_features):
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    X_train[numerical_features] = scaler.fit_transform(
        X_train[numerical_features]
    )

    # Transform test data using same scaler
    X_test[numerical_features] = scaler.transform(
        X_test[numerical_features]
    )

    return X_train, X_test, scaler


def one_hot_encode_data(X_train, X_test, one_hot_features):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[one_hot_features]
    )

    # Transform test data
    test_encoded = encoder.transform(
        X_test[one_hot_features]
    )

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(
        one_hot_features
    )

    # Convert to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    # Remove original categorical columns
    X_train = X_train.drop(
        columns=one_hot_features
    )

    X_test = X_test.drop(
        columns=one_hot_features
    )

    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )

    return X_train, X_test, encoder


def ordinal_encode_data(X_train, X_test, ordinal_features):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[ordinal_features]
    )

    # Transform test data
    test_encoded = encoder.transform(
        X_test[ordinal_features]
    )

    # Convert to DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=ordinal_features,
        index=X_train.index
    )

    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=ordinal_features,
        index=X_test.index
    )

    # Remove original ordinal columns
    X_train = X_train.drop(
        columns=ordinal_features
    )

    X_test = X_test.drop(
        columns=ordinal_features
    )

    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )

    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )

    return X_train, X_test, encoder


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------------

    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)


    # ---------------------------------------------------------
    # Split Dataset
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(df)

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)


    # ---------------------------------------------------------
    # Identify Features
    # ---------------------------------------------------------

    numerical_features, categorical_features = identify_features(
        X_train
    )

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)


    # ---------------------------------------------------------
    # Define Encoding Features
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # Handle Missing Values
    # ---------------------------------------------------------

    X_train, X_test, imputer = handle_missing_values(
        X_train,
        X_test,
        numerical_features
    )

    print(X_train[numerical_features].isnull().sum())

    print("\nMissing Value Handling completed.")


    # ---------------------------------------------------------
    # Standardization
    # ---------------------------------------------------------

    X_train, X_test, scaler = standardize_data(
        X_train,
        X_test,
        numerical_features
    )

    print("\nStandardization completed.")


    # ---------------------------------------------------------
    # One-Hot Encoding
    # ---------------------------------------------------------

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
    )

    print("One-Hot Encoding completed.")


    # ---------------------------------------------------------
    # Ordinal Encoding
    # ---------------------------------------------------------

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
    )

    print("Ordinal Encoding completed.")


    # ---------------------------------------------------------
    # Add Target Column
    # ---------------------------------------------------------

    X_train["PlacementStatus"] = y_train
    X_test["PlacementStatus"] = y_test


    # ---------------------------------------------------------
    # Create Data Directory
    # ---------------------------------------------------------

    data_path = r"C:\Users\scs87\PycharmProjects\Placementpredictsystem\data"

    os.makedirs(data_path, exist_ok=True)


    # ---------------------------------------------------------
    # Save Preprocessed Data
    # ---------------------------------------------------------

    train_file = os.path.join(
        data_path,
        "preprocessed_train.csv"
    )

    test_file = os.path.join(
        data_path,
        "preprocessed_test.csv"
    )

    X_train.to_csv(
        train_file,
        index=False
    )

    X_test.to_csv(
        test_file,
        index=False
    )


    # ---------------------------------------------------------
    # Completion Message
    # ---------------------------------------------------------

    print("\n----------------------------------------")
    print("PREPROCESSING COMPLETED")
    print("----------------------------------------")

    print("\nFinal Training Shape:")
    print(X_train.shape)

    print("\nFinal Testing Shape:")
    print(X_test.shape)

    print("\nPreprocessed Training Data:")
    print(X_train.head())

    print("\nPreprocessed Testing Data:")
    print(X_test.head())

    print("\nFiles saved successfully:")
    print(train_file)
    print(test_file)