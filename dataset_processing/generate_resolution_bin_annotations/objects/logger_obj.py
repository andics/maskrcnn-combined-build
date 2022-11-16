import logging
import os
import time

class Logger():
    def __init__(self, logger_name, logs_subdir, log_file_name, utils_helper):

        self.logger_name = logger_name
        self.logs_path = logs_subdir
        self.log_file_name = log_file_name
        self.utils_helper = utils_helper
        self.setup_logger()

    def setup_logger(self):
        if self.utils_helper.check_dir_and_make_if_na(self.logs_path):
            print("General log dir exists; proceeding...")
        else:
            print("General log dir did not exist; created one!")

        logger_name = self.logger_name
        logger_main = logging.getLogger(logger_name)

        self.log_file_current = os.path.join(self.logs_path, self.log_file_name + "_" +
                                             time.strftime("%Y_%m_%d_%H-%M-%S") + '.log')
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


        logger = logging.getLogger(self.logger_name)

        logger.info(message)
        print(message)