import sys
import os

from utils import utils_gen

class fileSystem:
    def __init__(self, _main_file_dir, unprocessed_file_path, config_file, logger):
        self._main_file_dir = _main_file_dir
        self.unprocessed_file_path = unprocessed_file_path
        self.config_file = config_file
        self.logger = logger
        #Assign value in case making a file copy in the unprocessed directory is set to false
        self.unprocessed_file_in_unprocessed_dir = None

    def setup_file_structure(self):
        self.general_unprocessed_files_folder = os.path.join(self._main_file_dir,
                                                  self.config_file["TRANSCRIPTION"]["unprocessed_files_dir"])
        self.general_processed_files_folder = os.path.join(self._main_file_dir,
                                                  self.config_file["TRANSCRIPTION"]["processed_files_dir"])

        if not utils_gen.check_dir_and_make_if_na(self.general_unprocessed_files_folder):
            self.print_("Created general unprocessed documents folder...")
        if not utils_gen.check_dir_and_make_if_na(self.general_processed_files_folder):
            self.print_("Created general processed documents folder...")

        self.unprocessed_file_name, self.unprocessed_file_ext = utils_gen.extract_filename_and_ext(self.unprocessed_file_path)
        self.processed_file_folder = os.path.join(self.general_processed_files_folder,
                                                  self.unprocessed_file_name)
        if not utils_gen.check_dir_and_make_if_na(self.processed_file_folder):
            self.print_("Created specific document processed folder. Making a file copy...")

        self.unprocessed_file_in_processed_dir = os.path.join(self.processed_file_folder,
                                                              self.unprocessed_file_name + "." + self.unprocessed_file_ext)
        utils_gen.copy_file_from_to(self.unprocessed_file_path, self.unprocessed_file_in_processed_dir)

        # Also make a copy in the unprocessed files directory - for cohesion & ease of access
        if self.config_file["TRANSCRIPTION"]["make_file_copy"]:
            self.unprocessed_file_in_unprocessed_dir = os.path.join(self.general_unprocessed_files_folder,
                                                                  self.unprocessed_file_name + "." + self.unprocessed_file_ext)
            utils_gen.copy_file_from_to(self.unprocessed_file_path, self.unprocessed_file_in_unprocessed_dir)
            self.print_("Created a file copy in unprocessed directory")

        return self.unprocessed_file_in_processed_dir, self.processed_file_folder,\
               self.unprocessed_file_in_unprocessed_dir, self.general_unprocessed_files_folder, \
               self.unprocessed_file_name, self.unprocessed_file_ext


    def print_(self, *messages):
        self.logger.log(*messages)