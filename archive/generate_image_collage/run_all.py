import sys
sys.path.append('/home/projects/bagon/andreyg/Projects/Variable_Resolution/Programming/maskrcnn_exp')

import os

from . import generate_n_image_predictions as generate_n_image_predictions
from . import generate_gt_visualizations as generate_gt_visualizations
from . import modify_file_names as modify_file_names
from . import generate_final_image_collage as generate_final_image_collage

def main():
    #NOTE: This runs everything EXCEPT generating predictions from the Multi_stacked
    #That has to be ran separately
    universal_plot_save_path = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Experiment_visualization/comparative_visualization_min_5"
    universal_preserve_original_dir_content = True

    generate_n_image_predictions.generate_n_image_predictions(universal_plot_save_path=universal_plot_save_path,
                                                              universal_preserve_original_dir_content=universal_preserve_original_dir_content)
    generate_gt_visualizations.generate_gt_visualizations(os.path.join(universal_plot_save_path, "Ground_truth"))
    modify_file_names.modify_file_names(folder_to_scan=universal_plot_save_path)
    generate_final_image_collage.generate_final_image_collage(general_folder_with_models_predictions=universal_plot_save_path,
                                                              folder_to_save_prediction_collage=os.path.join(universal_plot_save_path, "Final_collage"))



if __name__=="__main__":
    main()