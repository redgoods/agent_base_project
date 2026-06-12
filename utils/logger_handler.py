"""
日志处理模块
提供统一的日志记录功能，支持控制台输出和文件存储

日志级别（从低到高）：
- DEBUG: 调试信息，详细的诊断信息
- INFO: 一般信息，确认程序按预期运行
- WARNING: 警告信息，表示发生了意外情况，但程序仍能继续
- ERROR: 错误信息，表示发生了严重问题，程序无法执行某些功能
- CRITICAL: 严重错误，表示发生了严重问题，程序可能无法继续运行

"""
import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime

# 日志保存的根目录
# 所有日志文件都会保存在这个目录下
LOG_ROOT = get_abs_path("logs")

# 确保日志目录存在
# 如果目录不存在，则创建它（包括父目录）
# exist_ok=True 表示如果目录已存在，不会报错
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式配置
# 定义统一的日志输出格式，包含时间、名称、级别、文件位置和消息
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# 格式说明：
# %(asctime)s: 时间戳，例如：2026-06-10 14:30:45,123
# %(name)s: 日志器名称，例如：agent
# %(levelname)s: 日志级别，例如：INFO、ERROR、DEBUG
# %(filename)s: 产生日志的文件名，例如：main.py
# %(lineno)d: 代码行号，例如：42
# %(message)s: 日志消息内容


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    """
    获取配置好的日志记录器

    这是日志系统的核心函数，用于创建和配置日志记录器。

    Args:
        name: 日志器名称，用于区分不同的日志记录器
              默认为"agent"，可以根据模块名自定义，例如："rag"、"vector_db"等
        console_level: 控制台输出日志级别，默认为INFO
                      只有>=此级别的日志才会显示在控制台
        file_level: 文件保存日志级别，默认为DEBUG
                   只有>=此级别的日志才会写入日志文件
        log_file: 自定义日志文件路径，默认为None（自动生成）
                 如果指定，则使用指定的文件路径

    Returns:
        logging.Logger: 配置好的日志记录器对象
    """
    # 获取或创建日志器
    # 如果同名的日志器已存在，则返回已存在的日志器
    # 这样可以避免重复创建日志器
    logger = logging.getLogger(name)

    # 设置日志器的最低级别为DEBUG
    # 这意味着日志器会处理所有级别的日志消息
    logger.setLevel(logging.DEBUG)

    # 检查日志器是否已经有处理器（Handler）
    # 如果已有处理器，直接返回，避免重复配置
    # 这很重要，因为重复配置会导致日志消息被重复输出
    if logger.handlers:
        return logger

    # 配置控制台处理器（StreamHandler）
    # 用于将日志输出到控制台/终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)  # 设置控制台的日志级别
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)  # 设置输出格式

    # 将控制台处理器添加到日志器
    logger.addHandler(console_handler)

    # 配置文件处理器（FileHandler）
    # 用于将日志写入文件
    if not log_file:
        # 如果没有指定日志文件路径，则自动生成
        # 文件命名规则：{日志器名称}_{日期}.log
        # 例如：agent_20260610.log
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    # 创建文件处理器，使用UTF-8编码（支持中文）
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)  # 设置文件的日志级别
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)  # 设置输出格式

    # 将文件处理器添加到日志器
    logger.addHandler(file_handler)

    # 返回配置好的日志器
    return logger


# 快捷获取日志器
# 创建一个全局的默认日志器，名称为"agent"
# 其他模块可以直接导入使用：from utils.logger_handler import logger
logger = get_logger()


if __name__ == '__main__':
    """
    测试代码：验证日志系统功能

    运行此文件时，会测试不同级别的日志输出
    """
    # INFO级别：一般信息
    # 会在控制台和文件中显示
    logger.info("信息日志")

    # ERROR级别：错误信息
    # 会在控制台和文件中显示
    logger.error("错误日志")

    # WARNING级别：警告信息
    # 会在控制台和文件中显示
    logger.warning("警告日志")

    # DEBUG级别：调试信息
    # 只会在文件中显示（因为控制台级别设为INFO）
    logger.debug("调试日志")
