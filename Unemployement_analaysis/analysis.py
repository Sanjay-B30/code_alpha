import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")


# --------------------------------------------------
# Create Output Folder
# --------------------------------------------------

def create_output_folder():
    os.makedirs("output", exist_ok=True)


# --------------------------------------------------
# Load Both Datasets
# --------------------------------------------------

def load_datasets():

    dataset1 = pd.read_csv("dataset/Unemployment in India.csv")

    dataset2 = pd.read_csv("dataset/Unemployment_Rate_upto_11_2020.csv")

    print("\nDatasets Loaded Successfully!")

    return dataset1, dataset2


# --------------------------------------------------
# Clean Dataset
# --------------------------------------------------

def clean_data(df):

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Rename columns
    rename_columns = {
        "Region": "State",
        "Estimated Unemployment Rate (%)": "Unemployment Rate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)": "Labour Participation Rate"
    }

    df.rename(columns=rename_columns, inplace=True)

    # Convert Date column
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month_name()

    return df


# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------

def dataset_summary(df, title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistics:")
    print(df.describe(include="all"))


# --------------------------------------------------
# Save Cleaned Dataset
# --------------------------------------------------

def save_cleaned_data(df, filename):

    df.to_csv(filename, index=False)

    print(f"\nSaved : {filename}")

    # --------------------------------------------------
# Histogram
# --------------------------------------------------

def histogram(df, name):

    plt.figure(figsize=(8,5))

    sns.histplot(df["Unemployment Rate"], bins=20, kde=True, color="steelblue")

    plt.title("Distribution of Unemployment Rate")

    plt.xlabel("Unemployment Rate (%)")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(f"output/{name}_histogram.png")

    plt.close()


# --------------------------------------------------
# Average State-wise Unemployment
# --------------------------------------------------

def state_analysis(df, name):

    state = df.groupby("State")["Unemployment Rate"].mean().sort_values()

    plt.figure(figsize=(12,8))

    state.plot(kind="barh", color="green")

    plt.title("Average Unemployment Rate by State")

    plt.xlabel("Unemployment Rate (%)")

    plt.tight_layout()

    plt.savefig(f"output/{name}_state_analysis.png")

    plt.close()


# --------------------------------------------------
# Monthly Trend
# --------------------------------------------------

def monthly_trend(df, name):

    if "Month" not in df.columns:
        return

    order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    monthly = df.groupby("Month")["Unemployment Rate"].mean().reindex(order)

    plt.figure(figsize=(10,5))

    plt.plot(monthly.index, monthly.values, marker="o")

    plt.xticks(rotation=45)

    plt.title("Monthly Unemployment Trend")

    plt.xlabel("Month")

    plt.ylabel("Average Rate")

    plt.tight_layout()

    plt.savefig(f"output/{name}_monthly_trend.png")

    plt.close()


# --------------------------------------------------
# Top 10 States
# --------------------------------------------------

def top_states(df, name):

    top = df.groupby("State")["Unemployment Rate"].mean()

    top = top.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,6))

    sns.barplot(x=top.values, y=top.index)

    plt.title("Top 10 States by Unemployment Rate")

    plt.tight_layout()

    plt.savefig(f"output/{name}_top10.png")

    plt.close()


# --------------------------------------------------
# Bottom 10 States
# --------------------------------------------------

def bottom_states(df, name):

    bottom = df.groupby("State")["Unemployment Rate"].mean()

    bottom = bottom.sort_values().head(10)

    plt.figure(figsize=(10,6))

    sns.barplot(x=bottom.values, y=bottom.index)

    plt.title("Bottom 10 States by Unemployment Rate")

    plt.tight_layout()

    plt.savefig(f"output/{name}_bottom10.png")

    plt.close()


# --------------------------------------------------
# Urban vs Rural
# --------------------------------------------------

def urban_rural(df, name):

    if "Area" not in df.columns:
        return

    plt.figure(figsize=(7,5))

    sns.boxplot(
        data=df,
        x="Area",
        y="Unemployment Rate"
    )

    plt.title("Urban vs Rural Unemployment")

    plt.tight_layout()

    plt.savefig(f"output/{name}_urban_rural.png")

    plt.close()

    # --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------

def correlation_heatmap(df, name):

    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(8,6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(f"output/{name}_heatmap.png")

    plt.close()


# --------------------------------------------------
# Scatter Plot
# --------------------------------------------------

def scatter_plot(df, name):

    if "Area" not in df.columns:
        return

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        data=df,
        x="Employed",
        y="Unemployment Rate",
        hue="Area"
    )

    plt.title("Employment vs Unemployment")

    plt.tight_layout()

    plt.savefig(f"output/{name}_scatter.png")

    plt.close()


# --------------------------------------------------
# Pie Chart
# --------------------------------------------------

def pie_chart(df, name):

    if "Area" not in df.columns:
        return

    area = df.groupby("Area")["Employed"].sum()

    plt.figure(figsize=(6,6))

    plt.pie(
        area,
        labels=area.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Employment Share by Area")

    plt.tight_layout()

    plt.savefig(f"output/{name}_pie.png")

    plt.close()


# --------------------------------------------------
# COVID-19 Trend
# --------------------------------------------------

def covid_trend(df, name):

    if "Year" not in df.columns:
        return

    covid = df[df["Year"] == 2020]

    plt.figure(figsize=(12,6))

    plt.plot(
        covid["Date"],
        covid["Unemployment Rate"],
        marker="o"
    )

    plt.title("COVID-19 Unemployment Trend")

    plt.xlabel("Date")

    plt.ylabel("Unemployment Rate")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(f"output/{name}_covid.png")

    plt.close()


# --------------------------------------------------
# Generate All Charts
# --------------------------------------------------

def visualize_dataset(df, name):

    print(f"\nGenerating charts for {name}...")

    histogram(df, name)
    state_analysis(df, name)
    monthly_trend(df, name)
    top_states(df, name)
    bottom_states(df, name)
    urban_rural(df, name)
    correlation_heatmap(df, name)
    scatter_plot(df, name)
    pie_chart(df, name)
    covid_trend(df, name)

    print("Charts generated successfully.")