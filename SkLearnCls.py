from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class SkLearnCls:
    def machine_learning(self, df):
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


    def identify_products_to_purchase(self, df):
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
