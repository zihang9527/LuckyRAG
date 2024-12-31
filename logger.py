import logging
import datetime

def get_now_str():
    now = datetime.datetime.now()
    now_s = now.strftime("%Y-%m-%d %H:%M:%S")
    return now_s

def get_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # 创建文件处理器，并设置自定义格式
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = get_logger('logger', './log/log.txt')