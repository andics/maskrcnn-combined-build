import logging
import os
import time

class Logger():
    def __init__(self, config_file, utils_helper):
        self.main_logs_storage_path = config_file["LOGGING"]["main_logs_dir"]
        self.config_file = config_file

        self.logs_path = os.path.join(self.main_logs_storage_path, config_file["LOGGING"]["logs_subdir"])
        self.utils_helper = utils_helper
        self.setup_logger()

    def setup_logger(self):
        if self.utils_helper.check_dir_and_make_if_na(self.logs_path):
            print("General log dir exists; proceeding...")
        else:
            print("General log dir did not exist; created one!")

        logger_name = self.config_file["LOGGING"]["main_logger_name"]
        logger_main = logging.getLogger(logger_name)

        self.log_file_current = os.path.join(self.logs_path, 'log_' + time.strftime("%Y_%m_%d_%H-%M-%S") + '.log')
        logs_gen_file_handler = logging.FileHandler(self.log_file_current)

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        logs_gen_file_handler.setFormatter(formatter)
        logger_main.addHandler(logs_gen_file_handler)
        logger_main.setLevel(logging.INFO)

        return logger_name


    def log(self, *args):
        message = ""

        if type(args) == tuple:
            message = "".join(map(str, args))

        else:
            for _message_tmp in args:
                message = message + str(_message_tmp)


        logger = logging.getLogger(self.config_file["LOGGING"]["main_logger_name"])

        logger.info(message)
        print(message)