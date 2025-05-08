import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('sales_data.csv')

# ------------calculate total sales ------------
region_sales = df.groupby('Region')['Total'].sum()

#  ------------Set the style to dark------------
plt.style.use('dark_background')

#------------Plotting the bar chart------------
plt.figure(figsize=(10, 6))
region_sales.plot(kind='bar', color='red')

# ------------Adding title and labels------------
plt.title('Total Sales by Region', color='white')
plt.xlabel('Region', color='white')
plt.ylabel('Total Sales', color='white')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')

#------------ Save the plot------------
plt.savefig('total_sales_by_region.png')


plt.close()

#------------ Output------------
print("Bar chart saved as 'total_sales_by_region.png'.")


