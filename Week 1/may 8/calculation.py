import pandas as pd
import numpy as np

# 1. Load the CSV file
df = pd.read_csv('sales_data.csv')

# 2. Display the first 5 rows
print("HEAD of the DataFrame:")
print(df.head(), "\n")

# 3. Check for missing values
print("Missing Values Count:")
print(df.isnull().sum(), "\n")

# 4. Display summary statistics
print("Summary Statistics:")
print(df.describe(), "\n")

# 5. Calculate total sales by Region
print("Total Sales by Region:")
print(df.groupby('Region')['Total'].sum(), "\n")

# 6. Filter rows where Total sales > 2000
print("Sales Greater than 2000:")
print(df[df['Total'] > 2000], "\n")

# 7. Apply a 10% discount if Quantity is greater than 10
df['Discount'] = np.where(df['Quantity'] > 10, df['Total'] * 0.10, 0)
print("Products with Discounts Applied:")
print(df[['Product', 'Quantity', 'Total', 'Discount']], "\n")

# 8. Create a pivot table for sales by Region and Product
print("Pivot Table - Sales by Region and Product:")
pivot_table = pd.pivot_table(df, values='Total', index='Region', columns='Product', aggfunc='sum', fill_value=0)
print(pivot_table, "\n")
