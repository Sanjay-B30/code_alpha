from analysis import *

def main():
    print("=" * 60)
    print("     UNEMPLOYMENT ANALYSIS USING PYTHON")
    print("=" * 60)

    # Load datasets
    df1, df2 = load_datasets()

    # Clean datasets
    df1 = clean_data(df1)
    df2 = clean_data(df2)

    # Dataset information
    dataset_summary(df1, "Dataset 1 - Unemployment in India")
    dataset_summary(df2, "Dataset 2 - Unemployment Rate upto Nov 2020")

    # Create output folder
    create_output_folder()

    # Generate charts
    visualize_dataset(df1, "dataset1")
    visualize_dataset(df2, "dataset2")

    # Save cleaned data
    save_cleaned_data(df1, "output/Cleaned_Dataset1.csv")
    save_cleaned_data(df2, "output/Cleaned_Dataset2.csv")

    print("\nProject Completed Successfully!")
    print("Check the 'output' folder for charts and cleaned datasets.")

if __name__ == "__main__":
    main()