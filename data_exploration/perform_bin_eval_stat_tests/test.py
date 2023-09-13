import pandas as pd
from io import StringIO


filepath = "q:\\Projects\\Variable_resolution\\Programming\\maskrcnn-combined-build\\EXPERIMENTS/bin_eval_per_obj_type_ann_norm_small_med/evaluations/equiconst_pretrained_resnet_norm_0.1_340_an_ylarge_nfp/eval_across_bins_on_104.csv"
# Initialize lists to hold DataFrames and row data
dataframes = []
rowData = []

# Function to check whether a line is empty
def is_empty_line(line):
    return all(c in (',', ' ') for c in line)

# Open the csv and read line by line
with open(filepath, 'r') as file:
    for line in file:
        stripped_line = line.strip()

        # Check if the line is 'empty'
        if is_empty_line(stripped_line) or stripped_line == '':
            if rowData:
                df = pd.read_csv(StringIO("\n".join(rowData)), header=None)
                dataframes.append(df)
                rowData = []  # Clear the row data
        else:
            rowData.append(stripped_line)
            
# Add last batch of data to dataframe list, if it's non-empty
if rowData:
    df = pd.read_csv(StringIO("\n".join(rowData)), header=None)
    dataframes.append(df)

# Now, dataframes list has all your dataframes
for i, df in enumerate(dataframes):
    print(f"DataFrame #{i+1}:")
    print(df)
    print()