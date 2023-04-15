import os
import shutil
import logging

class recyclerObj:
    def __init__(self, prev_trial_folder, current_folder, files_to_recycle, current_trial_number):
        self.prev_trial_folder = prev_trial_folder
        self.current_folder = current_folder
        self.files_to_recycle = files_to_recycle
        self.current_trial_number = current_trial_number

    def copy_subfolders(self):
        for subdir, _, files in os.walk(self.prev_trial_folder):
            if not files:
                continue
            subfolder_name = os.path.basename(subdir)
            new_subdir = os.path.join(self.current_folder, subfolder_name)
            os.makedirs(new_subdir, exist_ok=True)
            for f in self.files_to_recycle:
                if f in files:
                    src_file = os.path.join(subdir, f)
                    dst_file = os.path.join(new_subdir, f)
                    if not os.path.exists(dst_file):
                        logging.info(f"Recycling file {src_file}")
                        shutil.copy(src_file, dst_file)
                    else:
                        logging.info(f"File {src_file} already present in Trial #{self.current_trial_number}, not recycling ...")
