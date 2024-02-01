import logging
import os

class StreamFileLogger():
    def __init__(self, name, log_dir=None):
        # set logger's format
        fmt = '%(asctime)s | %(message)s'
        datefmt='%Y/%m/%d %H:%M:%S'
        fmt = logging.Formatter(fmt, datefmt)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # screen
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)

        # file 
        if log_dir is not None:
            fh = logging.FileHandler(os.path.join(log_dir,name+'.log'), 'w')
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    def record(self, message):
        self.logger.info(message)