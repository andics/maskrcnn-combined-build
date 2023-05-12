import seaborn
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from data_exploration.plot_bin_perf_differences.utils.util_functions import Utilities_helper

class modelDifferencePlotter:
    """Plot the difference in model performance across all metrics for two models

        Notes
        ----------
        model_1_perf - model_2_perf will be taken as the difference value. Adjust setting models 1 and 2 accordingly

        Parameters
        ----------
        """

    _CSV_NAMING_PATTERN = "eval_across_bins_on_*.csv"
    _BEGINNING_OF_CSV_METRICS_COLUMN_INDEX = 3
    _END_OF_CSV_METRICS_COLUMN_INDEX_REVERS = -4

     # How much extra will be added to the max/min values
    _SCALER_MAX_MULT_FACTOR_ADD = 0.1
    _SCALER_MIN_MULT_FACTOR_ADD = 0.1

    def __init__(self, model_one_avg_perf_csv_path, model_two_avg_perf_csv_path, \
                 model_one_marg_perf_csv_path, model_two_marg_perf_csv_path,
                 marg_avg_perf_diff_csv_path, perf_diff_curves_file_path,
                 model_one_eval_folder, model_two_eval_folder,
                 utils_helper, scaler_file):

        self.model_one_avg_perf_csv_path = model_one_avg_perf_csv_path
        self.model_one_marg_perf_csv_path = model_one_marg_perf_csv_path

        self.model_two_avg_perf_csv_path = model_two_avg_perf_csv_path
        self.model_two_marg_perf_csv_path = model_two_marg_perf_csv_path

        self.marg_avg_perf_diff_csv_path = marg_avg_perf_diff_csv_path
        self.perf_diff_curve_path = perf_diff_curves_file_path
        self.model_one_eval_folder = model_one_eval_folder
        self.model_two_eval_folder = model_two_eval_folder
        self.utils_helper = utils_helper
        self.scaler_file = scaler_file

        self._s = modelDifferencePlotter._BEGINNING_OF_CSV_METRICS_COLUMN_INDEX
        self._e = modelDifferencePlotter._END_OF_CSV_METRICS_COLUMN_INDEX_REVERS

    def setup_objects_and_find_files(self):
        self.csv_files_model_1 = glob.glob(os.path.join(self.model_one_eval_folder, modelDifferencePlotter._CSV_NAMING_PATTERN))
        self.csv_files_model_2 = glob.glob(os.path.join(self.model_two_eval_folder, modelDifferencePlotter._CSV_NAMING_PATTERN))

        if not (len(self.csv_files_model_1) == 1 and len(self.csv_files_model_2) == 1):
            return False
        self.csv_files_model_1 = self.csv_files_model_1[0]
        self.csv_files_model_2 = self.csv_files_model_2[0]

        return True

    def read_scaler_info(self):
        # read the column range data from the .json file
        with open(self.scaler_file, 'r') as f:
            self.column_ranges = json.load(f)['columns']

    def create_avg_marg_and_diff_dataframes(self):
        df_model_1_avg_metrics_per_bin_across_trials = self.create_average_value_dataframes(self.csv_files_model_1)
        self.utils_helper.write_pd_dataframe_to_csv(df_model_1_avg_metrics_per_bin_across_trials,
                                                    self.model_one_avg_perf_csv_path)
        df_model_1_marg_gains_of_avg_df = self.create_marginal_gain_dataframe(df_model_1_avg_metrics_per_bin_across_trials)
        self.utils_helper.write_pd_dataframe_to_csv(df_model_1_marg_gains_of_avg_df,
                                                    self.model_one_marg_perf_csv_path)

        df_model_2_avg_metrics_per_bin_across_trials = self.create_average_value_dataframes(self.csv_files_model_2)
        self.utils_helper.write_pd_dataframe_to_csv(df_model_2_avg_metrics_per_bin_across_trials,
                                                    self.model_one_avg_perf_csv_path)
        df_model_2_marg_gains_of_avg_df = self.create_marginal_gain_dataframe(df_model_2_avg_metrics_per_bin_across_trials)
        self.utils_helper.write_pd_dataframe_to_csv(df_model_2_marg_gains_of_avg_df,
                                                    self.model_two_marg_perf_csv_path)

        self.marg_perf_difference_df = self.calculate_dataframe_difference(df_model_1_marg_gains_of_avg_df,
                                                                      df_model_2_marg_gains_of_avg_df)
        self.utils_helper.write_pd_dataframe_to_csv(self.marg_perf_difference_df, self.marg_avg_perf_diff_csv_path)


    def calculate_dataframe_difference(self, df_model_1: pd.DataFrame, df_model_2: pd.DataFrame) -> pd.DataFrame:
        # create a new data frame that is a copy of df1
        df_diff = df_model_1.copy()

        # subtract df2 from df1 for columns 4 to 10
        df_diff.iloc[:, self._s:self._e] =\
            df_model_1.iloc[:, self._s:self._e].sub(df_model_2.iloc[:, self._s:self._e])

        return df_diff


    def create_average_value_dataframes(self, csv_file_path):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Filter out non-numeric rows (excluding the header)
        df = df[pd.to_numeric(df['lower_bin_thresh'], errors='coerce').notnull()]

        # Group the rows by the first column and compute the mean of each group
        unique_bin_ids = df.lower_bin_thresh.unique()

        all_trials_avg_metrics_df = pd.DataFrame(columns = df.columns)
        for bin_id in unique_bin_ids:
            trial_rows_for_bin_id = df.loc[df['lower_bin_thresh'] == bin_id]
            trial_rows_for_bin_id_metrics = \
                trial_rows_for_bin_id.iloc[:, self._s:].apply(pd.to_numeric, errors = 'raise')
            trial_rows_for_bin_id_metrics_avg = trial_rows_for_bin_id_metrics.mean(axis = 0)
            row_to_append = trial_rows_for_bin_id.iloc[0].copy()
            row_to_append.iloc[self._s:] =\
                trial_rows_for_bin_id_metrics_avg

            all_trials_avg_metrics_df = all_trials_avg_metrics_df.append(row_to_append)

        print(all_trials_avg_metrics_df)
        return all_trials_avg_metrics_df


    def create_marginal_gain_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select only the numeric columns
        numeric_cols = df.iloc[:, self._s:self._e].select_dtypes(include='number').columns

        # Compute the marginal changes for each column
        marginal_changes = df[numeric_cols].diff()

        # Replace the first row with zeros, since we cannot calculate marginal gain for datapoint 0
        marginal_changes.iloc[0] = 0

        # Combine the marginal changes with the non-numeric columns
        df.iloc[:, self._s:self._e] = marginal_changes

        return df


    def generate_results_graph_photo(self, eval_across_bins_graph_file_path, differnce_df,
                                     use_scaler = True):
        # This function takes the generated .csv file and outputs a photo of the model performance graph
        if os.path.exists(eval_across_bins_graph_file_path):
            print("Graph file with eval across bins already exists!")
            return
        else:
            print("Generating graph photo ...")

        data = differnce_df
        column_names_metrics = list(data.columns)[-28:-4]
        bar_chart_columns = list(data.columns)[-4:]

        # create a 7x4 grid of plots
        fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(16, 28))

        # iterate over the grid of plots and plot each pair of columns
        for i, ax in enumerate(axs.flat):
            # extract the x and y columns for this plot
            x_col = f'lower_bin_thresh'
            x_data = data[x_col].values.astype(float)
            if i < len(column_names_metrics):
                y_col = column_names_metrics[i]

                y_data = data[y_col].values

                # compute the rolling average with window of 2
                rolling_avg = data[y_col].rolling(window=2, min_periods=2).mean()

                # plot the data on the current subplot
                ax.plot(x_data, y_data, marker='o', linestyle='--')
                ax.plot(x_data[1:], rolling_avg[1:], '--', linewidth=1.5, color='lightcoral', label='Moving Avg.')

                # set the title to the name of the y column
                ax.set_title(y_col)

                # set the y-axis limits to the min and max values for the current column
                if use_scaler:
                    # get the min and max values for the current column and scale for display
                    y_min = self.column_ranges[y_col]['min'] - \
                            modelDifferencePlotter._SCALER_MIN_MULT_FACTOR_ADD * abs(
                        float(self.column_ranges[y_col]['min']))
                    y_max = self.column_ranges[y_col]['max'] + \
                            modelDifferencePlotter._SCALER_MAX_MULT_FACTOR_ADD * abs(
                        float(self.column_ranges[y_col]['max']))
                    y_min = max(-1, y_min)
                    y_max = min(1, y_max)
                    ax.set_ylim([y_min, y_max])

                # hide the x and y-axis labels and ticks
                ax.set_xlabel('Bins (lower-thresh)')
                ax.set_ylabel(f'{y_col}')
                ax.set_title('')
            else:
                # plot the data on the current subplot as bar charts
                y_col = bar_chart_columns[i - len(column_names_metrics)]
                y_data = data[y_col].values
                ax.bar(x_data, y_data, width=0.05)

                # set the title to the name of the y column
                ax.set_title(y_col)

                # set the y-axis ticks to show the range of bar heights
                max_height = int(np.ceil(y_data.max()))
                min_height = int(np.floor(y_data.min()))
                num_ticks = 5

                y_ticks = np.asarray(self.utils_helper.generate_equispaced_numbers(min_height,
                                                                                   max_height,
                                                                                   num_ticks))
                ax.set_yticks(y_ticks)

                # hide the x-axis ticks and labels
                ax.set_xlabel('Bins (lower-thresh)')
                ax.set_ylabel(f'{y_col}')
                ax.set_title('')

        # adjust the layout of the subplots
        fig.tight_layout()

        # save the figure to a file
        fig.savefig(eval_across_bins_graph_file_path, dpi=300)
        print(f"Finished generating plot image!")
