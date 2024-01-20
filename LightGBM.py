from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


class LightGBM:
    def machine_learning(self, df):
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


    def identify_products_to_purchase(self, df):
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
