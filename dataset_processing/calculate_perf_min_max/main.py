import os
import csv
import json

parent_folder =
storage_location =

def find_csv_files_and_extract_info(parent_folder, storage_location):
    # Initialize a dictionary to store the min and max values for each column
    column_ranges = {}

    # Find all subfolders in parent_folder
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for subfolder in subfolders:
        # Look for .csv files in the subfolder
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv') and f.startswith('eval_across_bins_on_')]

        for csv_file in csv_files:
            # Extract the column values from the .csv file
            with open(os.path.join(subfolder, csv_file), 'r') as f:
                reader = csv.DictReader(f)
                for column in reader.fieldnames[3:-3]:
                    # If this is the first time we see this column, initialize its range
                    if column not in column_ranges:
                        column_ranges[column] = {'min': float('inf'), 'max': float('-inf')}

                    # Update the range for this column
                    for row in reader:
                        value = float(row[column])
                        column_ranges[column]['min'] = min(column_ranges[column]['min'], value)
                        column_ranges[column]['max'] = max(column_ranges[column]['max'], value)

    # Store the column ranges in a nested dictionary and save it to a .json file
    data = {'columns': {}}
    for column, range_info in column_ranges.items():
        data['columns'][column] = range_info

    with open(os.path.join(storage_location, 'column_ranges.json'), 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    find_csv_files_and_extract_info(parent_folder=parent_folder, storage_location=storage_location)