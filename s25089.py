import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def describe_data(data):
    print(data.info())
    print(data.describe())

    missing_values = data.isnull().sum()
    print("Brakujące wartości w danych:")
    print(missing_values)

    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    print("Kolumny numeryczne:")
    print(num_cols)
    print("Kolumny kategoryczne:")
    print(cat_cols)

    print(data[num_cols].describe())
    print(data[cat_cols].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(data['score'], kde=True)
    plt.title('Rozkład wyników testu')
    #plt.savefig('score_histogram.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(data['gender'])
    plt.title('Rozkład płci')
    #plt.savefig('gender_countplot.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(data['income'])
    plt.title('Rozkład zarobków')
    #plt.savefig('income_countplot.png')
    plt.show()

def preprocess_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Kolumna '{col}': {mapping}")
        df[col] = le.transform(df[col])

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('score')
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    return train_set, test_set

def train_linear_regression(train_set, test_set):
    X_train = train_set.drop(columns=['score','rownames'])
    y_train = train_set['score']
    X_test = test_set.drop(columns=['score','rownames'])
    y_test = test_set['score']

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Szczegółowy raport oceny modelu:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")
    print(f"R-squared (R2): {r2:.2f}")

def main():
    url = 'https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
    df = pd.read_csv(url)
    describe_data(df)
    train_set, test_set = preprocess_data(df)
    train_linear_regression(train_set, test_set)

if __name__ == "__main__":
    main()
