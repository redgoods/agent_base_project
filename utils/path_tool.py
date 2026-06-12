"""
路径工具模块
为整个工程提供统一的绝对路径管理

在Python项目中，相对路径的使用会因为当前工作目录的变化而产生问题，
此模块通过基于项目根目录的路径解析，确保路径的一致性和可靠性。

主要功能：
- 获取项目根目录
- 将相对路径转换为绝对路径
- 避免因工作目录变化导致的路径问题
"""
import os


def get_project_root() -> str:
    """
    获取工程项目所在的根目录

    工作原理：
    1. 获取当前文件的绝对路径
    2. 获取当前文件所在目录的父目录（即项目根目录）

    目录结构示例：
    agent 项目/
    ├── utils/
    │   ├── __init__.py
    │   └── path_tool.py  ← 当前文件
    ├── config/
    ├── logs/
    └── main.py

    Args:
        无

    Returns:
        str: 项目根目录的绝对路径
    """
    # 获取当前文件的绝对路径
    # __file__ 是Python内置变量，表示当前文件的路径
    current_file = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    # dirname()函数去除路径中的文件名，只保留目录部分
    current_dir = os.path.dirname(current_file)

    # 获取项目根目录（utils目录的父目录）
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    将相对路径转换为基于项目根目录的绝对路径

    工作原理：
    1. 获取项目根目录
    2. 将相对路径与项目根目录拼接
    3. 返回完整的绝对路径

    Args:
        relative_path: 相对路径（相对于项目根目录）
                      例如："config/rag.yml"、"logs/app.log"

    Returns:
        str: 完整的绝对路径
    """
    # 获取项目根目录
    project_root = get_project_root()

    # 使用os.path.join拼接路径，自动处理路径分隔符
    # 在Windows上使用"\"，在Linux/Mac上使用"/"
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    """
    测试代码：验证路径工具功能

    运行此文件时，会测试相对路径转换为绝对路径的功能
    """
    # 测试配置文件的绝对路径
    test_path = get_abs_path("config/config.txt")
    print(f"测试路径: {test_path}")

    # 验证路径是否正确
    print(f"路径存在: {os.path.exists(test_path)}")
    print(f"路径类型: {'目录' if os.path.isdir(test_path) else '文件' if os.path.isfile(test_path) else '不存在'}")
