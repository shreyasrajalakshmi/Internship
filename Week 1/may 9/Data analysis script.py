import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1:
def create_sample_data():
    if not os.path.exists("data.csv"):
        data = {
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Name": [
                "John Doe", "Jane Smith", "Emily Johnson", "Michael Brown",
                "Emma Wilson", "Liam Davis", "Olivia Martinez",
                "Noah Thomas", "Ava White", "Mason Lee"
            ],
            "Age": [23, 25, 22, 24, 21, 27, 24, 28, 20, 23],
            "Gender": ["Male", "Female", "Female", "Male", "Female",
                       "Male", "Female", "Male", "Female", "Male"],
            "Income": [50000, 60000, 55000, 62000, 47000,
                       71000, 69000, 48000, 52000, 58000]
        }
        df = pd.DataFrame(data)
        df.to_csv("data.csv", index=False)
        print("data.csv file created successfully.")
    else:
        print("data.csv already exists.")

create_sample_data()

# ____________Base Class____________
class DataAnalyzer:
    def __init__(self, file_path):  
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load data from CSV into DataFrame"""
        try:
            self.df = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except FileNotFoundError:
            print("Error: CSV file not found.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def show_summary(self):
        """Display basic statistics"""
        if self.df is not None:
            print("\n--- Data Summary ---")
            print(self.df.describe())
        else:
            print("Data not loaded.")

    def calculate_statistics(self):
        """Return mean and median of Age and Income"""
        if self.df is not None:
            mean_age = self.df['Age'].mean()
            median_income = self.df['Income'].median()  
            print(f"\nMean Age: {mean_age}")
            print(f"Median Income: {median_income}")
        else:
            print("Data not loaded.")

#  ____________Subclass with additional student-specific methods____________

class StudentDataAnalyzer(DataAnalyzer):
    def filter_by_gender(self, gender):
        """Return filtered data by gender"""
        if self.df is not None:
            filtered = self.df[self.df['Gender'].str.lower() == gender.lower()]
            print(f"\nFiltered Data (Gender = {gender}):\n", filtered)
        else:
            print("Data not loaded.")

    def plot_distribution(self):
        """Plot age and income distribution"""
        if self.df is not None:
            plt.figure(figsize=(12, 5))

            # ____________Age Distribution____________
            
            plt.subplot(1, 2, 1)
            plt.hist(self.df['Age'], color='skyblue', edgecolor='black')
            plt.title("Age Distribution")
            plt.xlabel("Age")
            plt.ylabel("Count")

            # ____________Income Distribution____________
            
            plt.subplot(1, 2, 2)
            plt.hist(self.df['Income'], color='lightgreen', edgecolor='black')
            plt.title("Income Distribution")
            plt.xlabel("Income")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.show()
        else:
            print("Data not loaded.")

# _____________Main function____________

def main():
    analyzer = StudentDataAnalyzer("data.csv")
    analyzer.load_data()
    analyzer.show_summary()
    analyzer.calculate_statistics()
    analyzer.filter_by_gender("Female")
    analyzer.plot_distribution()

if __name__ == "__main__":  # Fixed block
    main()
