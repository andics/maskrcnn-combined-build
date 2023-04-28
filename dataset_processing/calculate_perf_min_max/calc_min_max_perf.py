import os
import csv
import json

parent_folder = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/EXPERIMENTS/bin_eval_per_obj_type_ann_norm_small_med/evaluations"
storage_location = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/dataset_processing/calculate_perf_min_max/perf_no_full_res_ylarge_28.04"

def find_csv_files_and_extract_info(parent_folder, storage_location):
    # Define the columns of interest
    columns_of_interest = ['bbox_AP', 'bbox_AP50', 'bbox_AP75', 'bbox_APs', 'bbox_APm', 'bbox_APl',
                           'bbox_AR@1', 'bbox_AR@10', 'bbox_AR', 'bbox_ARs', 'bbox_ARm', 'bbox_ARl',
                           'segm_AP', 'segm_AP50', 'segm_AP75', 'segm_APs', 'segm_APm', 'segm_APl',
                           'segm_AR@1', 'segm_AR@10', 'segm_AR', 'segm_ARs', 'segm_ARm', 'segm_ARl']

    # Initialize a dictionary to store the min and max values for each column
    column_ranges = {column: {'min': float('inf'), 'max': float('-inf')} for column in columns_of_interest}

    # Find all subfolders in parent_folder
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

    for subfolder in subfolders:
        # Look for .csv files in the subfolder
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv') and f.startswith('eval_across_bins_on_')]

        print(csv_files)
        assert len(csv_files) <= 1

        for csv_file in csv_files:
            # Extract the column values from the .csv file
            with open(os.path.join(subfolder, csv_file), 'r') as f:
                reader = csv.DictReader(f)
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