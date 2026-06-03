import numpy as np
import pandas as pd
import sys

import os
from pathlib import Path

# Set working directory to the folder where CleanDataPreliminary.py lives
project_root = Path(__file__).resolve().parent.parent

print("Working directory set to:", project_root)

dataset = pd.read_csv(project_root / 'Raw data from SVP' / '20200611DailyCountsOfPlateletReturnsAndDeliveries.txt', delimiter=';')

# using only relevant columns and rows
dataset_filtered = dataset[dataset['class'] != 'return']
dataset_filtered = dataset_filtered.drop(columns=['class'])

# replace all where n="1-5" with an integer uniformly sampled between 1 and 5 (uniform distribution)
def replace_n(row):
    if row['n'] == '1-5':
        return np.random.randint(1, 6)
    else:
        return int(row['n'])

# set seed for reproducibility
np.random.seed(43)
for index, row in dataset_filtered.iterrows():
    dataset_filtered.at[index, 'n'] = replace_n(row)

# changing data types
dataset_filtered['date'] = pd.to_datetime(dataset_filtered['date'], format='%Y-%m-%d')
dataset_filtered['n'] = dataset_filtered['n'].astype(int)
dataset_filtered['size'] = dataset_filtered['size'].astype('category')
dataset_filtered['name'] = dataset_filtered['name'].astype('category')

# make a list of unique hospital names
unique_names = dataset_filtered['name'].unique().tolist()
# create IDs
hospital_ids = pd.DataFrame({
    'Hospital Name': unique_names,
    'Hospital ID': range(1, len(unique_names) + 1)
})
hospital_ids.to_csv(project_root / 'hospital_ids.csv', index=False)
# join the IDs to data
dataset_filtered = dataset_filtered.merge(hospital_ids, left_on='name', right_on='Hospital Name', how='left')
dataset_filtered = dataset_filtered.drop(columns=['Hospital Name'])

# import csv file again with locations
hospital_ids = pd.read_csv(project_root / 'Input Files' / 'finland_hospitals_with_regions.csv', sep=';')
# cleaning
hospital_ids['y_lat'] = hospital_ids['y_lat'].str.replace(',', '.').astype(float)
hospital_ids['x_lon'] = hospital_ids['x_lon'].str.replace(',', '.').astype(float)

# join location data and clean
dataset_filtered = dataset_filtered.merge(hospital_ids, left_on='Hospital ID', right_on='hospital_id', how='left')
dataset_filtered = dataset_filtered.drop(columns=['hospital_id', 'hospital_name'])

# adding holidays
holidays= pd.read_csv(project_root / 'Input Files' / 'finnish_holidays_2012_2021.csv', sep=',')
dataset_filtered['is_holiday'] = dataset_filtered['date'].isin(pd.to_datetime(holidays['date'])).astype(int)
