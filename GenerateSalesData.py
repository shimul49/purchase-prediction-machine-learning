import pandas as pd
import numpy as np
from faker import Faker
import random

class GenerateSalesData:
    def generate_sales_data(self, num_records, num_products):
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