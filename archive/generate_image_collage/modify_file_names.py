from utils.util_functions import Utilities_helper as ut

import os

def modify_file_names(folder_to_scan = "/home/projects/bagon/andreyg/Projects/Variable_Resolution/Experiment_visualization/comparative_visualization_min_5"):
    #Desired file name length without the file extension
    desired_file_name_lenght = 9

    util_helper = ut()

    collective_folders = util_helper.gather_subfolders(folder_to_scan)
    print("Gathered the following subfolders: ")
    [print(folder) for folder in collective_folders]

    for folder in collective_folders:
        current_subdir_name = os.path.basename(os.path.normpath(folder))
        current_folder_files_wo_paths = util_helper.gather_subfiles(folder)

        total_num_images = len(current_folder_files_wo_paths)
        #print("Found the following files: ", current_folder_files_wo_paths)
        print("Number of files found in folder {}: ".format(current_subdir_name), total_num_images)

        current_folder_sample_element = current_folder_files_wo_paths[0]
        file_name, file_extension = os.path.splitext(current_folder_sample_element)
        num_to_reduce_file_name_length = len(file_name) - desired_file_name_lenght

        print("Length of image names without extensions: ", len(file_name))

        if num_to_reduce_file_name_length < 0:
            raise Exception("Files is already less than desired length")

        counter = 0

        for file in current_folder_files_wo_paths:
            file_current_name, file_current_ext = os.path.splitext(file)

            file_new_name = file_current_name[num_to_reduce_file_name_length:]
            file_new_name_w_extension = file_new_name + file_current_ext
            file_new_path = os.path.join(folder, file_new_name_w_extension)

            file_old_path = os.path.join(folder, file)

            os.rename(file_old_path, file_new_path)
            print("Image {} renamed into {}".format(file_current_name, file_new_name))
            counter += 1
            print("Progress ", counter, " / ", total_num_images)

        print("Finished with folder {}".format(current_subdir_name))


if __name__=="__main__":
    modify_file_names()