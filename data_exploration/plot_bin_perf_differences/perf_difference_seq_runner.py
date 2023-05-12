import seaborn
import os
import sys
import argparse
import re

from pathlib import Path

try:
    path_main = str(Path(os.path.dirname(os.path.realpath(__file__))).parents[1])
    print(path_main)
    sys.path.append(path_main)
    os.chdir(path_main)
    sys.path.remove('/workspace/object_detection')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("Environmental paths updated successfully!")
except Exception:
    print("Tried to edit environmental paths but was unsuccessful!")

from data_exploration.plot_bin_perf_differences.utils.util_functions import Utilities_helper
from data_exploration.plot_bin_perf_differences.objects.plotter_obj import modelDifferencePlotter

class flowRunner:
    #First string is model prefix, second is bin high-res area size
    _MODEL_AVG_PERF_CSV_FILE_NAME_PATTERN = "%s_avg_pref_%s.csv"
    _MODEL_MARG_PERF_CSV_FILE_NAME_PATTERN = "%s_marg_of_avg_pref_%s.csv"
    _PERF_DIFF_CSV_FILE_NAME_PATTERN = "perf_diff_of_marg_of_avg_%s.csv"
    _PERF_DIFF_PNG_FILE_NAME_PATTERN = "perf_diff_of_marg_of_avg_%s.png"

    def __init__(self):
        parser = argparse.ArgumentParser(description='Potential arguments for complete resolution-bin evaluation pipeline')
        parser.add_argument('-mop', '--model-one-prefix', nargs='?',
                            type=str,
                            default = "var",
                            required = False,
                            help='The folder prefix which will be searched for to count model 1')
        parser.add_argument('-mtp', '--model-two-prefix', nargs='?',
                            type=str,
                            default = "equiconst",
                            required = False,
                            help='The folder prefix which will be searched for to count model 2')
        parser.add_argument('-ged', '--general-eval-dir', nargs='?',
                            required=False,
                            type = str,
                            default="w:/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_combined/EXPERIMENTS/bin_eval_per_obj_type_ann_norm_small_med/evaluations",
                            help='The general directory in which all experiments (for all sizes) are stored')
        parser.add_argument('-pcsd', '--plots-n-csv-store-dir', nargs='?',
                            required=False,
                            type = str,
                            default="Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/data_exploration/plot_bin_perf_differences/plots_and_csv",
                            help='The general directory in which all perf. difference curves will be stored')
        parser.add_argument('-sf', '--scaler-file', nargs='?',
                            required=False,
                            type=str,
                            default="Q:/Projects/Variable_resolution/Programming/maskrcnn-combined-build/dataset_processing/calculate_marg_perf_diff_min_max/marg_avg_perf_diff_var_equiconst_12.05/column_ranges.json",
                            help='The file that will be used to create boundaries on each plot. Needs to be created after generating the'
                                 'plots and csv-s at least once, so as to se the different values they exhibit on the plots')

        self.args = parser.parse_args()
        self.model_one_prefix = self.args.model_one_prefix
        self.model_two_prefix = self.args.model_two_prefix
        self.general_eval_dir = self.args.general_eval_dir
        self.plots_n_csv_store_dir = self.args.plots_n_csv_store_dir
        self.scaler_file = self.args.scaler_file


    def setup_objects_and_file_structure(self):
        self.utils_helper = Utilities_helper()
        self.comparison_storage_folder = os.path.join(self.plots_n_csv_store_dir,
                                                      self.model_one_prefix + "_" + self.model_two_prefix)
        self.utils_helper.check_dir_and_make_if_na(self.comparison_storage_folder)


    def create_model_folders_pairs_n_storage(self):
        # create a list to store the matching folder names
        self.model_one_avg_perf_csv_paths = []
        self.model_one_marg_perf_csv_paths = []

        self.model_two_avg_perf_csv_paths = []
        self.model_two_marg_perf_csv_paths = []

        self.marg_avg_perf_diff_csv_paths = []
        self.perf_diff_curves_file_paths = []
        self.m1_m2_exp_folder_pairs = []

        self.hr_area_sizes = []

        # regular expression pattern to match an integer in a string
        int_pattern = re.compile(r"(?<!\.)(?<!\d)(-?\d+)(?!\d|\.)")

        # traverse the directory tree and find folders with prefixes "var" and "equiconst"
        root, subdirs, _ = next(os.walk(self.general_eval_dir))
        for model_one_dir_name in subdirs:
            if model_one_dir_name.startswith(self.model_one_prefix):
                # extract the integer value from the folder name
                try:
                    m1_name_number = int(int_pattern.findall(model_one_dir_name)[0])
                except Exception as e:
                    print(f"Folder {model_one_dir_name} name did not contain integers. Skipping...")
                    continue

                # look for a matching folder with prefix "var" and same integer value
                for model_two_dir_name in subdirs:
                    if model_two_dir_name.startswith(self.model_two_prefix):
                        m2_name_number = int(int_pattern.findall(model_two_dir_name)[0])

                        if m2_name_number == m1_name_number:
                            # add tuple with the names of the two folders to the matching_folders list
                            model_one_avg_perf_csv = os.path.join(self.comparison_storage_folder,
                                                                  flowRunner._MODEL_AVG_PERF_CSV_FILE_NAME_PATTERN %
                                                                  (self.model_one_prefix, str(m1_name_number)))
                            model_one_marg_perf_csv = os.path.join(self.comparison_storage_folder,
                                                                  flowRunner._MODEL_MARG_PERF_CSV_FILE_NAME_PATTERN %
                                                                  (self.model_one_prefix, str(m1_name_number)))

                            model_two_avg_perf_csv = os.path.join(self.comparison_storage_folder,
                                                                  flowRunner._MODEL_AVG_PERF_CSV_FILE_NAME_PATTERN %
                                                                  (self.model_two_prefix, str(m1_name_number)))
                            model_two_marg_perf_csv = os.path.join(self.comparison_storage_folder,
                                                                  flowRunner._MODEL_MARG_PERF_CSV_FILE_NAME_PATTERN %
                                                                  (self.model_two_prefix, str(m1_name_number)))

                            model_diff_avg_perf_csv = os.path.join(self.comparison_storage_folder,
                                                                   flowRunner._PERF_DIFF_CSV_FILE_NAME_PATTERN % str(m1_name_number))
                            model_diff_avg_perf_png = os.path.join(self.comparison_storage_folder,
                                                                   flowRunner._PERF_DIFF_PNG_FILE_NAME_PATTERN % str(m1_name_number))

                            self.model_one_avg_perf_csv_paths.append(model_one_avg_perf_csv)
                            self.model_one_marg_perf_csv_paths.append(model_one_marg_perf_csv)

                            self.model_two_avg_perf_csv_paths.append(model_two_avg_perf_csv)
                            self.model_two_marg_perf_csv_paths.append(model_two_marg_perf_csv)

                            self.marg_avg_perf_diff_csv_paths.append(model_diff_avg_perf_csv)
                            self.perf_diff_curves_file_paths.append(model_diff_avg_perf_png)

                            self.m1_m2_exp_folder_pairs.append((os.path.join(root, model_one_dir_name), os.path.join(root,
                                                                                                                     model_two_dir_name)))
                            self.hr_area_sizes.append(str(m1_name_number))

    def run_comparison_for_all(self):

        for model_one_avg_perf_csv_path, model_two_avg_perf_csv_path, \
            model_one_marg_perf_csv_path, model_two_marg_perf_csv_path, \
            marg_avg_perf_diff_csv_path, perf_diff_curves_file_path, \
            m1_m2_exp_folder_pair, hr_area in zip(self.model_one_avg_perf_csv_paths,
                                           self.model_two_avg_perf_csv_paths,
                                           self.model_one_marg_perf_csv_paths,
                                           self.model_two_marg_perf_csv_paths,
                                           self.marg_avg_perf_diff_csv_paths,
                                           self.perf_diff_curves_file_paths,
                                           self.m1_m2_exp_folder_pairs,
                                           self.hr_area_sizes):

            print(f"Working on {hr_area} for {self.model_one_prefix}-{self.model_two_prefix}")
            if not os.path.exists(perf_diff_curves_file_path):
                _model_difference_plotter = modelDifferencePlotter(model_one_avg_perf_csv_path = model_one_avg_perf_csv_path,
                                                                   model_two_avg_perf_csv_path = model_two_avg_perf_csv_path,
                                                                   model_one_marg_perf_csv_path = model_one_marg_perf_csv_path,
                                                                   model_two_marg_perf_csv_path = model_two_marg_perf_csv_path,
                                                                   marg_avg_perf_diff_csv_path = marg_avg_perf_diff_csv_path,
                                                                   perf_diff_curves_file_path = perf_diff_curves_file_path,
                                                                   model_one_eval_folder = m1_m2_exp_folder_pair[0],
                                                                   model_two_eval_folder = m1_m2_exp_folder_pair[1],
                                                                   utils_helper = self.utils_helper,
                                                                   scaler_file = self.scaler_file)
                if _model_difference_plotter.setup_objects_and_find_files():
                    _model_difference_plotter.read_scaler_info()
                    _model_difference_plotter.create_avg_marg_and_diff_dataframes()
                    _model_difference_plotter.generate_results_graph_photo(_model_difference_plotter.perf_diff_curve_path,
                                                                           _model_difference_plotter.marg_perf_difference_df, \
                                                                           use_scaler = True)
                    print(f"Finished {hr_area}, moving to next (if any) ...")
                else:
                    print(f"Encountered problem with {hr_area} bin while finding _on_NUM.csv files. Moving to next (if any) ...")
            else:
                print(f"Graph photo exists. Moving to next bin (if any) ...")
                continue



if __name__ == "__main__":
    flow_runner = flowRunner()
    flow_runner.setup_objects_and_file_structure()
    flow_runner.create_model_folders_pairs_n_storage()
    flow_runner.run_comparison_for_all()
