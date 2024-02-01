# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''
import logging
import getpass
import sys

class MyLog(object):
    def __init__(self,init_file=None):
        user=getpass.getuser()
        self.logger=logging.getLogger(user)
        self.logger.setLevel(logging.DEBUG)
        if init_file==None:
            logFile=sys.argv[0][0:-3]+'.log'
        else:
            logFile=init_file
        formatter=logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')

        logHand=logging.FileHandler(logFile,encoding="utf8")
        logHand.setFormatter(formatter)
        logHand.setLevel(logging.INFO)#只记录错误

        logHandSt=logging.StreamHandler()
        logHandSt.setFormatter(formatter)

        self.logger.addHandler(logHand)
        self.logger.addHandler(logHandSt)

    def debug(self,msg):
        self.logger.debug(msg)
    def info(self,msg):
        self.logger.info(msg)
    def warn(self,msg):
        self.logger.warning(msg)
    def error(self,msg):
        self.logger.error(msg)
    def critical(self,msg):
        self.logger.critical(msg)

if __name__=='__main__':
    mylog=MyLog()
    mylog.debug("I'm debug")
    mylog.info("I'm info")
    mylog.warn("I'm warning")
    mylog.error("I'm error")
    mylog.critical("I'm critical")