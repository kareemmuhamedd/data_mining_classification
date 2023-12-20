import os
import sqlite3
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from classifier_functions import *
from utils import *


# load data from database 
def read_data_from_db(file_path):
    conn = sqlite3.connect(file_path)

    try:
        cursor = conn.cursor()

        # Retrieve the first table name from the sqlite_master table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        result = cursor.fetchone()

        if result is None:
            print("No tables found in the database.")
            conn.close()
            return None

        table_name = result[0]

        # Read the data from the retrieved table
        query = f"SELECT * FROM {table_name}"
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data

    except sqlite3.Error as e:
        print(f"An error occurred while reading the database: {e}")
        return None

# load data from any type
def load_data(path):
    try:
        _, file_extension = os.path.splitext(path)

        if file_extension == ".csv":
            data = pd.read_csv(path)
        elif file_extension in [".xls", ".xlsx"]:
            data = pd.read_excel(path)
        elif file_extension in [".db", ".sql"]:
            data = read_data_from_db(path)

        else:
            raise ValueError(
                "Unsupported file format. Please provide a CSV, Excel, SQL, or SQLite database file."
            )

        return data

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None

    except pd.errors.ParserError:
        print(
            "Error occurred while parsing the file. Please check if the file format is correct."
        )
        return None

    except Exception as e:
        print("An error occurred while loading the data:", str(e))
        return None


# prerpocessing data
def preprocess_data_for_visualization(data):
    # Identify column types
    categorical_columns = data.select_dtypes(include="object").columns
    numerical_columns = data.select_dtypes(include=["int", "float"]).columns

    # Handle missing values in numerical columns
    data[numerical_columns] = data[numerical_columns].fillna(
        data[numerical_columns].mean()
    )

    # Handle missing values in categorical columns
    data[categorical_columns] = data[categorical_columns].fillna(
        data[categorical_columns].mode().iloc[0]
    )

    # Encode categorical features
    encoded_data = pd.get_dummies(data, columns=categorical_columns)

    # Scale numerical features
    scaled_data = (
        encoded_data[numerical_columns] - encoded_data[numerical_columns].mean()
    ) / encoded_data[numerical_columns].std()
    encoded_data[numerical_columns] = scaled_data

    # Rename columns with camel case
    encoded_data.columns = encoded_data.columns.str.replace("_", "")

    return encoded_data

def preprocess_data_for_classifier(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # inplace=True like i re assigne df = df.replace....
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Handling missing values for numerical columns with mean
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # Handling missing values for categorical columns with mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

def preprocess_data_for_classifier(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # inplace=True like i re assigne df = df.replace....
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Handling missing values for numerical columns with mean
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # Handling missing values for categorical columns with mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])


# Determine features and goal
def determine_features_and_goal(df):
    features = df.drop(columns=['class'])
    goal = df['class']
    return features, goal


# Split data into training and testing
def split_data(features, goal, test_size=0.5, random_state=3):
    X_train, X_test, y_train, y_test = train_test_split(
        features, goal, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test



# Label encode categorical features
def label_encode_categorical_features(df):
    le = LabelEncoder()
    # fit_transform convert categorical data to 0,1 if i have two feature of 1,2,3 if i have three ...
    df = df.apply(le.fit_transform)
    return df


# Calculate performance using confusion matrix
def calculate_performance(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    Accuracy = round(metrics.accuracy_score(y_test, y_pred) * 100)
    Precision = round(metrics.precision_score(y_test, y_pred,
              average='macro', zero_division=1) * 100)
    Recall = round(metrics.recall_score(y_test, y_pred,
              average='macro', zero_division=1) * 100)
    return Accuracy, Precision, Recall, cm










