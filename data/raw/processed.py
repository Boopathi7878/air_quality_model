import pandas as pd
import os

# Set the file path to the raw data (Excel file)
filename = 'raw.xlsx'
file_path = os.path.join('data', 'raw', filename)

# Check if the file exists and load it
if os.path.exists(file_path):
    try:
        # Load the Excel file with the correct sheet name ('raw')
        df = pd.read_excel(file_path, sheet_name='raw')
        print(f"Loaded raw data from {filename}")
        
        # Check data types to understand columns
        print("Data types of columns:\n", df.dtypes)

        # Select only numeric columns for filling missing values
        numeric_columns = df.select_dtypes(include=['number']).columns
        print(f"Numeric columns: {numeric_columns}")

        # Fill missing values only for numeric columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        print("Data cleaned: Missing values filled with column mean.")

        # Save the cleaned data to the 'processed' folder
        processed_filename = 'cleaned_air_quality_data.csv'
        processed_file_path = os.path.join('data', 'processed', processed_filename)
        df.to_csv(processed_file_path, index=False)
        print(f"Processed data saved to {processed_filename}")

    except Exception as e:
        print(f"Error loading or processing the file {filename}: {e}")
else:
    print(f"The file {filename} does not exist in the 'raw' folder.")
