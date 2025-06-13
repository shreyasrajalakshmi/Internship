import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DataAnalyser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_csv()

    def load_csv(self):
        try:
            df = pd.read_csv(self.file_path)
            print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None

    def explore_data(self):
        print("\n=== DATA OVERVIEW ===")
        print("First 5 rows:")
        print(self.df.head())
        print("\nData types:")
        print(self.df.dtypes)
        print("\nSummary statistics:")
        print(self.df.describe().round(2))
        print("\nMissing values count:")
        print(self.df.isnull().sum())

    def clean_data(self):
        df = self.df.copy()
        old_shape = df.shape[0]
        df = df.drop_duplicates()
        print(f"Removed {old_shape - df.shape[0]} duplicate rows.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val:.2f}")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in '{col}' with mode: '{mode_val}'")
        print("Data cleaning completed.")
        self.df = df

    def analyze_data(self):
        print("\n=== DATA ANALYSIS ===")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\nNumerical data analysis:")
            for col in numeric_cols[:5]:
                print(f"\nColumn: {col}")
                print(f"  Mean: {self.df[col].mean():.2f}")
                print(f"  Median: {self.df[col].median():.2f}")
                print(f"  Std Dev: {self.df[col].std():.2f}")
                print(f"  Min: {self.df[col].min():.2f}")
                print(f"  Max: {self.df[col].max():.2f}")
        cat_cols = self.df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            print("\nCategorical data analysis:")
            for col in cat_cols[:3]:
                print(f"\nColumn: {col}")
                value_counts = self.df[col].value_counts().head(5)
                print("  Top 5 values:")
                for val, count in value_counts.items():
                    print(f"    {val}: {count} ({count/len(self.df)*100:.1f}%)")

    def create_visualizations(self, output_folder="output_plots"):
        print("\n=== CREATING VISUALIZATIONS ===")
        Path(output_folder).mkdir(exist_ok=True)
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        if len(numeric_cols) > 0:
            for i, col in enumerate(numeric_cols[:3]):
                plt.figure(figsize=(10, 6))
                counts, bins, _ = plt.hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                if len(df[col].dropna()) > 1:
                    mn, mx = plt.xlim()
                    plt.xlim(mn, mx)
                    kde_xs = np.linspace(mn, mx, 300)
                    kde_ys = np.histogram(df[col].dropna(), bins=bins, density=True)[0]
                    kde_ys = np.convolve(kde_ys, np.ones(5)/5, mode='same')
                    plt.plot(bins[:-1], kde_ys * np.max(counts) / np.max(kde_ys), color='darkblue', lw=2)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.tight_layout()
                filename = f"{output_folder}/histogram_{col}.png"
                plt.savefig(filename)
                plt.close()
                print(f"Saved histogram for '{col}' to {filename}")

        if len(cat_cols) > 0:
            for i, col in enumerate(cat_cols[:3]):
                plt.figure(figsize=(10, 6))
                top_categories = df[col].value_counts().head(10)
                bars = plt.bar(range(len(top_categories)), top_categories.values, color='lightblue', edgecolor='black')
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', ha='center', va='bottom')
                plt.title(f'Top Categories in {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(range(len(top_categories)), top_categories.index, rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                filename = f"{output_folder}/barplot_{col}.png"
                plt.savefig(filename)
                plt.close()
                print(f"Saved bar plot for '{col}' to {filename}")

        if len(numeric_cols) >= 2:
            plt.figure(figsize=(12, 10))
            corr_matrix = df[numeric_cols].corr()
            im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im)
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    text = plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", color="black")
            plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha='right')
            plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            filename = f"{output_folder}/correlation_matrix.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved correlation matrix to {filename}")

        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, edgecolor='w', s=50)
            plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            filename = f"{output_folder}/scatter_plot.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved scatter plot to {filename}")

        if len(numeric_cols) > 0:
            plt.figure(figsize=(12, 8))
            plt.boxplot([df[col].dropna() for col in numeric_cols[:5]], labels=[col for col in numeric_cols[:5]], patch_artist=True)
            plt.title('Box Plots')
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename = f"{output_folder}/box_plots.png"
            plt.savefig(filename)
            plt.close()
            print(f"Saved box plots to {filename}")


def main():
    file_path = input("Enter the path to the CSV file: ")
    analyzer = DataAnalyser(file_path)

    if analyzer.df is not None:
        while True:
            print("\nChoose an option:")
            print("1. Explore Data")
            print("2. Clean Data")
            print("3. Analyze Data")
            print("4. Create Visualizations")
            print("5. Exit")
            choice = input("Enter your choice (1-5): ")

            if choice == '1':
                analyzer.explore_data()
            elif choice == '2':
                analyzer.clean_data()
            elif choice == '3':
                analyzer.analyze_data()
            elif choice == '4':
                analyzer.create_visualizations()
            elif choice == '5':
                print("Exiting the tool.")
                break
            else:
                print("Invalid choice. Please try again.")
    else:
        print("Failed to load data. Exiting...")

if __name__ == "__main__":
    main()
