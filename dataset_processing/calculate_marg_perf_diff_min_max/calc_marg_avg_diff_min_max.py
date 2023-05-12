import os
import csv
import json

from pathlib import Path

parent_folder = "Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/data_exploration/plot_bin_perf_differences/plots_and_csv/var_full_res"
storage_location = "Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/dataset_processing/calculate_marg_perf_diff_min_max/marg_avg_perf_diff_var_full_res_12.05"

def find_csv_files_and_extract_info(parent_folder, storage_location):
    # Check if storage exists
    Path(storage_location).mkdir(exist_ok=True)

    # Define the columns of interest
    columns_of_interest = ['bbox_AP', 'bbox_AP50', 'bbox_AP75', 'bbox_APs', 'bbox_APm', 'bbox_APl',
                           'bbox_AR@1', 'bbox_AR@10', 'bbox_AR', 'bbox_ARs', 'bbox_ARm', 'bbox_ARl',
                           'segm_AP', 'segm_AP50', 'segm_AP75', 'segm_APs', 'segm_APm', 'segm_APl',
                           'segm_AR@1', 'segm_AR@10', 'segm_AR', 'segm_ARs', 'segm_ARm', 'segm_ARl']

    # Initialize a dictionary to store the min and max values for each column
    column_ranges = {column: {'min': float('inf'), 'max': float('-inf')} for column in columns_of_interest}

    for subfolder in [parent_folder]:
        # Look for .csv files in the subfolder
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv') and f.startswith('perf_diff_of_marg')]

        print(csv_files)

        for csv_file in csv_files:
            # Extract the column values from the .csv file
            with open(os.path.join(subfolder, csv_file), 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    for column in columns_of_interest:
                        try:
                            value = float(row[column])
                        except ValueError as e:
                            #We reached a row of spaces, separating trials
                            print(f"Failed to convert to float: {row[column]}")
                            continue

                        column_ranges[column]['min'] = min(column_ranges[column]['min'], value)
                        column_ranges[column]['max'] = max(column_ranges[column]['max'], value)

    # Store the column ranges in a nested dictionary and save it to a .json file
    data = {'columns': column_ranges}

    with open(os.path.join(storage_location, 'column_ranges.json'), 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    find_csv_files_and_extract_info(parent_folder=parent_folder, storage_location=storage_location)