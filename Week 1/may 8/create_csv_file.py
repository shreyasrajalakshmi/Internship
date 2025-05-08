import pandas as pd

#____________sample data____________
data = {
    'Date': pd.date_range(start='2024-01-01', periods=4, freq='D'),
    'Region': ['California', 'Texas', 'Florida', 'New York'],
    'Product': ['Rice', 'Wheat', 'Sugar', 'Oil'],
    'Quantity': [12, 18, 6, 10],
    'Unit Price': [110, 210, 160, 120]
}


df = pd.DataFrame(data)


df['Total'] = df['Quantity'] * df['Unit Price']

#____________Saving to CSV____________

df.to_csv('sales_data.csv', index=False)

print("CSV file 'sales_data.csv' created successfully.")
