import numpy as np
import pandas as pd
import sys

import os
from pathlib import Path

# Define the project root directory (one level above the current script)
project_root = Path(__file__).resolve().parent.parent

print("Working directory set to:", project_root)

# Load raw platelet delivery and return data
dataset = pd.read_csv(project_root / 'Raw data from SVP' / '20200611DailyCountsOfPlateletReturnsAndDeliveries.txt', delimiter=';')

# Keep only delivery records by removing return entries
dataset_filtered = dataset[dataset['class'] != 'return']
# Remove the now unnecessary class column
dataset_filtered = dataset_filtered.drop(columns=['class'])

# Replace entries where demand is recorded as "1-5"
# with a random integer between 1 and 5
def replace_n(row):
    if row['n'] == '1-5':
        return np.random.randint(1, 6)
    else:
        return int(row['n'])

# Set random seed to ensure reproducible results
np.random.seed(43)

# Apply replacement to all rows
for index, row in dataset_filtered.iterrows():
    dataset_filtered.at[index, 'n'] = replace_n(row)

# Convert columns to appropriate data types
dataset_filtered['date'] = pd.to_datetime(dataset_filtered['date'], format='%Y-%m-%d')
dataset_filtered['n'] = dataset_filtered['n'].astype(int)
dataset_filtered['size'] = dataset_filtered['size'].astype('category')
dataset_filtered['name'] = dataset_filtered['name'].astype('category')

# Create a list of unique hospital names
unique_names = dataset_filtered['name'].unique().tolist()
# Assign a unique integer ID to each hospital
hospital_ids = pd.DataFrame({
    'Hospital Name': unique_names,
    'Hospital ID': range(1, len(unique_names) + 1)
})

# Save hospital ID mapping for later use
hospital_ids.to_csv(project_root / 'Input Files' / 'hospital_ids.csv', index=False)
# Add hospital IDs to the main dataset
dataset_filtered = dataset_filtered.merge(hospital_ids, left_on='name', right_on='Hospital Name', how='left')
# Remove duplicate hospital name column created during merge
dataset_filtered = dataset_filtered.drop(columns=['Hospital Name'])

# Load hospital location and region information
hospital_ids = pd.read_csv(project_root / 'Input Files' / 'finland_hospitals_with_regions.csv', sep=';')

# Convert latitude and longitude from Finnish decimal format
# (comma separator) to floating-point numbers
hospital_ids['y_lat'] = hospital_ids['y_lat'].str.replace(',', '.').astype(float)
hospital_ids['x_lon'] = hospital_ids['x_lon'].str.replace(',', '.').astype(float)

# Merge location information into the dataset
dataset_filtered = dataset_filtered.merge(hospital_ids, left_on='Hospital ID', right_on='hospital_id', how='left')
# Remove redundant columns from the merge
dataset_filtered = dataset_filtered.drop(columns=['hospital_id', 'hospital_name'])

# Load Finnish public holiday dates
holidays= pd.read_csv(project_root / 'Input Files' / 'finnish_holidays_2012_2021.csv', sep=',')
# Create a binary holiday indicator
# 1 if the date is a public holiday, otherwise 0
dataset_filtered['is_holiday'] = dataset_filtered['date'].isin(pd.to_datetime(holidays['date'])).astype(int)
