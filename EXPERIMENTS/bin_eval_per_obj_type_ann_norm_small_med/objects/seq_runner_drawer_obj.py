import os
import typing as T
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pandas as pd

class seqRunnerDrawerObj:
    def __init__(self, logger, utils_helper):
        self.logger = logger
        self.utils_helper = utils_helper

    def create_combined_trials_csv(self, csv_files: T.List[str], path_to_save_combined_in: str):
        if os.path.exists(path_to_save_combined_in):
            self.logger.info("Combined CSV file with eval across bins already exists!")
            return

        # Initialize an empty list to hold the concatenated rows
        concatenated_rows = []

        # Loop through each CSV file and concatenate its rows
        for idx, csv_file in enumerate(csv_files):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                # Get the header row from the first CSV file
                if idx == 0:
                    header_row = next(reader)
                    concatenated_rows.append(header_row)
                # Add a row of "-" strings before the rows of the next CSV file are written
                else:
                    concatenated_rows.append([' ' for _ in header_row])
                # Append the rows to the concatenated_rows list
                for row in reader:
                    concatenated_rows.append(row)

        # Write the concatenated rows to a new CSV file
        with open(path_to_save_combined_in, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(concatenated_rows)

        self.logger.info("Generated combined CSV file ...")


    def generate_combined_results_graph_photo_plt(self, eval_across_bins_csv_file_paths: T.List[str],
                                                  eval_across_bins_graph_file_path: str):
        # This function takes the generated .csv files and outputs a photo of the model performance graph
        if os.path.exists(eval_across_bins_graph_file_path):
            self.logger.info("CSV file with eval across bins already exists!")
            return

        # load the column names from the first csv file
        data = pd.read_csv(eval_across_bins_csv_file_paths[0])
        column_names_metrics = list(data.columns)[-28:-4]
        bar_chart_columns = list(data.columns)[-4:]

        # create a 7x4 grid of plots
        fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(16, 28))

        # generate an arbitrary number of colors depending on the number of csv files
        num_files = len(eval_across_bins_csv_file_paths)
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, num_files))

        # iterate over the grid of plots and plot each pair of columns
        for i, ax in enumerate(axs.flat):
            # extract the x and y columns for this plot
            x_col = f'lower_bin_thresh'
            if i < len(column_names_metrics):
                y_col = column_names_metrics[i]

                all_x_data = []
                all_y_data = []
                # plot data from each csv file with a different color
                for j, csv_file_path in enumerate(eval_across_bins_csv_file_paths):
                    _data = pd.read_csv(csv_file_path)
                    x_data = _data[x_col].values
                    y_data = _data[y_col].values

                    # plot the data on the current subplot with a different color
                    ax.plot(x_data, y_data, marker='o', linewidth=1, linestyle='-', color=colors[j])
                    all_x_data.extend(x_data.tolist())
                    all_y_data.extend(y_data.tolist())

                # plot the line of best fit
                slope, intercept = np.polyfit(all_x_data, all_y_data, 1)
                line_of_best_fit_y = slope * np.array(all_x_data) + intercept
                ax.plot(all_x_data, line_of_best_fit_y, '-', linewidth=1.5, color='black', label='L.b.f.')

                # set the title to the name of the y column
                ax.set_title(y_col)

                # hide the x and y axis labels and ticks
                ax.set_xlabel('Bins (lower-thresh)')
                ax.set_ylabel(f'{y_col}')
                ax.set_title('')
            else:
                # plot the data on the current subplot as bar charts
                y_col = bar_chart_columns[i - len(column_names_metrics)]
                y_data = data[y_col].values
                x_data = data[x_col].values
                ax.bar(x_data, y_data, width=0.05)

                # set the title to the name of the y column
                ax.set_title(y_col)

                # set the y-axis ticks to show the range of bar heights
                max_height = int(np.ceil(y_data.max()))
                min_height = int(np.floor(y_data.min()))
                num_ticks = 5

                y_ticks = np.array(self.utils_helper.generate_equispaced_numbers(min_height,
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
        self.logger.info(f"Finished generating plot image!")

    def generate_combined_results_graph_photo_seaborn(self, eval_across_bins_csv_file_paths: T.List[str],
                                                  eval_across_bins_graph_file_path: str):
        # This function takes the generated .csv files and outputs a photo of the model performance graph
        if os.path.exists(eval_across_bins_graph_file_path):
            self.logger.info("CSV file with eval across bins already exists!")
            return

        # load the column names from the first csv file
        data = pd.read_csv(eval_across_bins_csv_file_paths[0])
        column_names_metrics = list(data.columns)[-28:-4]
        bar_chart_columns = list(data.columns)[-4:]

        # create a 7x4 grid of plots
        fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(16, 28))

        # generate an arbitrary number of colors depending on the number of csv files
        num_files = len(eval_across_bins_csv_file_paths)
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, num_files))

        # iterate over the grid of plots and plot each pair of columns
        for i, ax in enumerate(axs.flat):
            # extract the x and y columns for this plot
            x_col = f'lower_bin_thresh'
            if i < len(column_names_metrics):
                y_col = column_names_metrics[i]

                all_x_data = []
                all_y_data = []
                # collect data from each csv file into a single list
                for j, csv_file_path in enumerate(eval_across_bins_csv_file_paths):
                    _data = pd.read_csv(csv_file_path)
                    x_data = _data[x_col].values
                    y_data = _data[y_col].values

                    all_x_data.extend(x_data.tolist())
                    all_y_data.extend(y_data.tolist())

                # calculate the mean and standard deviation of the y-data at each unique x-value
                x_unique, y_mean, y_std = scipy.stats.binned_statistic(all_x_data, all_y_data, statistic='mean',
                                                                       bins=np.unique(all_x_data))

                # create a Seaborn lineplot of the mean values with error bounds
                sns.lineplot(x=x_unique, y=y_mean, ci='sd', ax=ax, color=colors[0])
                ax.fill_between(x_unique, y_mean - y_std, y_mean + y_std, alpha=0.2, color=colors[0])

                # set the title to the name of the y column
                ax.set_title(y_col)

                # hide the x and y axis labels and ticks
                ax.set_xlabel('Bins (lower-thresh)')
                ax.set_ylabel(f'{y_col}')
                ax.set_title('')
            else:
                # plot the data on the current subplot as bar charts
                y_col = bar_chart_columns[i - len(column_names_metrics)]
                y_data = data[y_col].values
                x_data = data[x_col].values
                ax.bar(x_data, y_data, width=0.05)

                # set the title to the name of the y column
                ax.set_title(y_col)

                # set the y-axis ticks to show the range of bar heights
                max_height = int(np.ceil(y_data.max()))
                min_height = int(np.floor(y_data.min()))
                num_ticks = 5

                y_ticks = np.array(self.utils_helper.generate_equispaced_numbers(min_height,
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
        self.logger.info(f"Finished generating plot image!")