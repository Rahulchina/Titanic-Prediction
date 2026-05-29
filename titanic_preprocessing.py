import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    # Put the Titanic dataset CSV in the workspace root or update this path.
    csv_path = os.path.join(os.path.dirname(__file__), 'Titanic-Dataset.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "Place Titanic-Dataset.csv in the repository root or update csv_path."
        )

    df = pd.read_csv(csv_path)
    print("First 5 rows:")
    print(df.head())
    print("\nData info:")
    print(df.info())
    print("\nMissing values by column:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe())

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop('Cabin', axis=1, inplace=True)

    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    print("\nAfter encoding:")
    print(df.head())

    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Fare'])
    plt.title('Fare Distribution with Outliers')
    plt.xlabel('Fare')
    plt.show()

    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

    scaler = StandardScaler()
    num_cols = ['Age', 'Fare']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print("\nFinal processed data sample:")
    print(df.head())
    print("\nFinal data info:")
    print(df.info())
    print("\nFinal missing values by column:")
    print(df.isnull().sum())
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == '__main__':
    main()
