import pandas as pd
import numpy as np

# Load data from CSV
file_path = (r"D:\\Codes\\Projects\\ML\\air_quality_model\data\\processed\\processed.csv")
df = pd.read_csv(file_path)
print("✅ Data loaded successfully!")

# Clean the data (handle NaN values)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Split into features (X) and target (y)
target_col = "AQI"
X = df.drop(columns=[target_col])
y = df[target_col]

# Save the cleaned data to a CSV
df.to_csv(r"D:\\Codes\\Projects\\ML\\air_quality_model\data\\processed\\cleaned_data.csv", index=False)
print("✅ Cleaned data saved successfully!")
