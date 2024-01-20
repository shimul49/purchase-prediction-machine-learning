from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


class XGBoost:
    def identify_products_to_purchase(self, df):
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
