import pandas as pd
import numpy as np
from faker import Faker
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from datetime import datetime
from xgboost import XGBClassifier


# np.random.seed(42)
# random.seed(42)


def generate_sales_data(num_records, num_products):
    fake = Faker()
    # products = [f'Product {i}' for i in range(1, num_products + 1)]
    products = [i for i in range(1, num_products + 1)]

    data = {
        'Invoice_No': [fake.uuid4() for _ in range(num_records)],
        'Product_ID': [f'P{i}' for i in range(1, num_products + 1) for _ in range(num_records // num_products)],
        'Product_Name': [random.choice(products) for _ in range(num_records)],
        'Quantity': [random.randint(1, 10) for _ in range(num_records)],
        'Price': [round(random.uniform(10, 100), 2) for _ in range(num_records)],
        'Discount': [round(random.uniform(0, 0.2), 2) for _ in range(num_records)],
        'Tax': [round(random.uniform(0, 0.1), 2) for _ in range(num_records)],
        'Total_Amount': [],
        'Invoice_Date': [fake.date_between('-1y', 'today') for _ in range(num_records)],
        'Remarks': [fake.sentence() for _ in range(num_records)],
    }

    # Calculate Total_Amount
    data['Total_Amount'] = [
        round((qty * price * (1 - discount) * (1 + tax)), 2) for qty, price, discount, tax in zip(
            data['Quantity'], data['Price'], data['Discount'], data['Tax']
        )
    ]

    df = pd.DataFrame(data)
    return df


def machine_learning(df):
    # Creating a binary target variable based on Quantity
    df['Demand_Level'] = df['Quantity'].apply(lambda x: 'High' if x >= 5 else 'Low')

    # Features and target variable
    features = df[['Quantity', 'Product_Name']]
    target = df['Demand_Level']

    # Use ColumnTransformer to apply different transformers to different columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('product_name', OneHotEncoder(sparse_output=False, drop='first'), ['Product_Name']),
            ('quantity', 'passthrough', ['Quantity'])
        ],
        remainder='drop'
    )

    # Fit and transform the features
    features_encoded = column_transformer.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    # Choose a machine learning algorithm (Random Forest as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))


def identify_products_to_purchase(df):
    # Creating a binary target variable based on Quantity
    df['Demand_Level'] = df['Quantity'].apply(lambda x: 'High' if x >= 5 else 'Low')

    # Features and target variable
    features = df[['Product_Name', 'Quantity']]
    target = df['Demand_Level']

    # Use ColumnTransformer to apply different transformers to different columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('product_name', OneHotEncoder(sparse_output=False, drop='first'), ['Product_Name']),
            ('quantity', 'passthrough', ['Quantity'])
        ],
        remainder='drop'
    )

    # Fit and transform the features
    features_encoded = column_transformer.fit_transform(features)

    # Convert feature names to strings
    feature_names = [str(col) for col in column_transformer.get_feature_names_out(['Product_Name', 'Quantity'])]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    # Choose a machine learning algorithm (Random Forest as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Identify products to purchase based on the model predictions
    df['Predicted_Demand_Level'] = model.predict(features_encoded)
    high_demand_products = df[df['Predicted_Demand_Level'] == 'High'][['Product_Name', 'Quantity']].groupby('Product_Name').agg({'Quantity': 'sum'})
    high_demand_products.reset_index(inplace=True)
    
    # Sort products by total sales quantity in descending order
    high_demand_products = high_demand_products.sort_values(by='Quantity', ascending=False)

    print("\nProducts to consider for purchase:")
    for index, row in high_demand_products.iterrows():
        print(f"{row['Product_Name']} ({row['Quantity']} sold)")


# LightGBM
def lgbm_machine_learning(df):
    # Creating a binary target variable based on Quantity
    df['Demand_Level'] = df['Quantity'].apply(lambda x: 1 if x >= 5 else 0)

    # Features and target variable
    features = df[['Quantity', 'Product_Name']]
    target = df['Demand_Level']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Choose a machine learning algorithm (LightGBM as an example)
    model = LGBMClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy}')

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))


def lgbm_identify_products_to_purchase(df):
    # Creating a binary target variable based on Quantity
    df['Demand_Level'] = df['Quantity'].apply(lambda x: 1 if x >= 5 else 0)

    # Convert 'Product_Name' to numerical labels using direct conversion to integers
    df['Product_Name'] = df['Product_Name'].astype(int)

    # Features and target variable
    features = df[['Quantity', 'Product_Name']]
    target = df['Demand_Level']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Choose a machine learning algorithm (LightGBM as an example)
    model = LGBMClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Identify products to purchase based on the model predictions
    df['Predicted_Demand_Level'] = model.predict(features)
    high_demand_products = df[df['Predicted_Demand_Level'] == 1][['Product_Name', 'Quantity']].groupby('Product_Name').agg({'Quantity': 'sum'})
    high_demand_products.reset_index(inplace=True)

    # Sort products by total sales quantity in descending order
    high_demand_products = high_demand_products.sort_values(by='Quantity', ascending=False)

    print("\nProducts to consider for purchase:")
    for index, row in high_demand_products.iterrows():
        print(f"{row['Product_Name']} ({row['Quantity']} sold)")


# XGBoost
def xgboost_identify_products_to_purchase(df):
    # Creating a binary target variable based on Quantity
    df['Demand_Level'] = df['Quantity'].apply(lambda x: 1 if x >= 5 else 0)

    # Features and target variable
    features = df[['Quantity', 'Product_Name']]
    target = df['Demand_Level']

    # Use ColumnTransformer to apply different transformers to different columns
    column_transformer = ColumnTransformer(
        transformers=[
            ('product_name', OneHotEncoder(sparse_output=False, drop='first'), ['Product_Name']),
            ('quantity', 'passthrough', ['Quantity'])
        ],
        remainder='drop'
    )

    # Fit and transform the features
    features_encoded = column_transformer.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    # Use LabelEncoder to encode the target variable
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Choose XGBoost as the machine learning algorithm
    model = XGBClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Identify products to purchase based on the model predictions
    df['Predicted_Demand_Level'] = model.predict(features_encoded)
    high_demand_products = df[df['Predicted_Demand_Level'] == 1][['Product_Name', 'Quantity']].groupby('Product_Name').agg({'Quantity': 'sum'})
    high_demand_products.reset_index(inplace=True)

    # Sort products by total sales quantity in descending order
    high_demand_products = high_demand_products.sort_values(by='Quantity', ascending=False)

    print("\nProducts to consider for purchase (XGBoost):")
    for index, row in high_demand_products.iterrows():
        print(f"{row['Product_Name']} ({row['Quantity']} sold)")


if __name__ == '__main__':
    df = generate_sales_data(100000, 10)
    
    # Export DataFrame to CSV
    # df.to_csv(Faker().uuid4() + '.csv', index=False)
    df.to_csv('random_sales_data.csv', index=False)
    print("CSV file exported successfully.")

    print("==========SKlearn prediction==========")
    sk_starttime = datetime.now()
    print("Start: " + str(sk_starttime))
    machine_learning(df)
    identify_products_to_purchase(df)
    sk_endtime = datetime.now()
    print("End: " + str(sk_endtime))
    time_required = sk_endtime - sk_starttime

    print("==========LightGBM Prediction==========")
    lgbm_starttime = datetime.now()
    print("Start: " + str(lgbm_starttime))
    lgbm_machine_learning(df)
    lgbm_identify_products_to_purchase(df)
    lgbm_endtime = datetime.now()
    print("End: " + str(lgbm_endtime))
    lgbm_time_required = lgbm_endtime - lgbm_starttime

    print("==========XGBoost Prediction==========")
    xgboost_starttime = datetime.now()
    print("Start: " + str(xgboost_starttime))
    xgboost_identify_products_to_purchase(df)
    xgboost_endtime = datetime.now()
    print("End: " + str(xgboost_endtime))
    xgboost_time_required = xgboost_endtime - xgboost_starttime

    print("==========Time Comparison==========")
    print("SKLearn Time required: " + str(time_required))
    print("LightGBM Time required: " + str(lgbm_time_required))
    print("XGBoost Time required: " + str(xgboost_time_required))
