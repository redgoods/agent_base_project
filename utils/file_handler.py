"""
文件处理工具模块
提供文件MD5计算、文件列表获取、PDF和文本文件加载等功能
主要用于RAG系统中文档的处理和管理
"""
import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def get_file_md5_hex(filepath: str):
    """
    计算文件的MD5哈希值（十六进制字符串）

    MD5（Message Digest Algorithm 5）是一种广泛使用的加密哈希函数，
    产生128位（16字节）的哈希值，通常以32位十六进制字符串表示。

    Args:
        filepath: 文件的绝对路径或相对路径

    Returns:
        str: 文件的MD5哈希值（32位十六进制字符串）
        None: 如果文件不存在、不是文件或计算失败则返回None
    """

    # 检查文件是否存在
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    # 检查路径是否指向文件（而非目录）
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    # 创建MD5哈希对象
    md5_obj = hashlib.md5()

    # 设置每次读取的块大小为4KB
    # 分块读取的原因：避免大文件一次性读取到内存导致内存溢出
    chunk_size = 4096

    try:
        # 以二进制模式打开文件
        with open(filepath, "rb") as f:
            # 使用海象运算符（:=）读取文件块，直到文件结束
            while chunk := f.read(chunk_size):
                # 将读取的块更新到MD5计算中
                md5_obj.update(chunk)
            """
            海象运算符（:=）是Python 3.8+的新特性
            等价的传统写法如下：
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            # 获取最终的MD5哈希值（十六进制字符串）
            md5_hex = md5_obj.hexdigest()
            return md5_hex

    except Exception as e:
        # 捕获并记录处理过程中的异常
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """
    列出指定目录下所有符合指定文件类型的文件

    Args:
        path: 要扫描的目录路径
        allowed_types: 允许的文件扩展名元组，例如：('.pdf', '.txt', '.md')

    Returns:
        tuple[str]: 符合条件的文件路径元组
        tuple[str]: 如果路径不是目录，返回allowed_types（错误情况）
    """
    files = []

    # 检查路径是否存在且是目录
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    # 遍历目录中的所有文件和子目录
    for f in os.listdir(path):
        # 检查文件是否以允许的扩展名结尾
        if f.endswith(allowed_types):
            # 构建完整路径并添加到结果列表
            files.append(os.path.join(path, f))

    # 将列表转换为元组返回（元组不可变，更安全）
    return tuple(files)

def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """
    加载PDF文件并将其转换为LangChain Document对象列表

    Args:
        filepath: PDF文件的路径
        passwd: PDF密码（如果有密码保护），默认为None

    Returns:
        list[Document]: 包含PDF内容的Document对象列表
        每个Document对象包含：
        - page_content: 页面文本内容
        - metadata: 元数据（包含页码、源文件路径等信息）
    示例：
        >>> docs = pdf_loader("document.pdf")
        >>> print(len(docs))  # 输出PDF的页数
        >>> print(docs[0].page_content)  # 输出第一页的文本
    """
    # 使用LangChain的PyPDFLoader加载PDF文件
    # load()方法返回Document对象列表
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    """
    加载文本文件并将其转换为LangChain Document对象列表

    Args:
        filepath: 文本文件的路径（支持.txt, .md, .py等各种文本格式）

    Returns:
        list[Document]: 包含文本内容的Document对象列表
        通常只有一个Document对象，包含整个文件的内容
    """
    # 使用LangChain的TextLoader加载文本文件
    # load()方法返回Document对象列表
    return TextLoader(filepath, encoding="utf-8").load()
