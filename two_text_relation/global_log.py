#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
'''
@Author   :   zxzjt
@Contact  :   zhujingtao@loonxi.com
@License  :   (C) Copyright 2013-2018, loonxi
@Software :   PyCharm Community Edition
@File     :   global_log.py
@Created  :   2018/7/26 20:19
@Version  :   0.0.1
@Desc     :   logger
'''
#-------------------------------------------------------------------------------

"""Code Base

"""

import logging

def logging_config(logger, path):
    """日志配置
    :param logger:创建Logging对象
    :return:None
    """
    # 指定logger输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s')
    # 文件日志
    file_handler = logging.FileHandler(path, encoding='utf8')
    file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
    # 控制台日志
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.formatter = formatter  # 也可以直接给formatter赋值
    # 为logger添加的日志处理器
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)
    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)
    return logger

log_enquiry_debug = 'enquiry_recomm_debug'
#log_enquiry_data = 'enquiry_recomm_data'
package_path = "./two_text_relation/"
# /usr/local/python_enquiry/enquiry_recommend_TwoTextRelation/two_text_relation/

log_path = 'log/'

logger_debug = logging_config(logger=logging.getLogger(log_enquiry_debug), path=package_path + log_path + log_enquiry_debug + '.log')
#logger_data = logging_config(logger=logging.getLogger(log_enquiry_data), path=package_path + log_path + log_enquiry_data + '.log')
