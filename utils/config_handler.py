"""
配置文件加载模块
用于加载和管理项目的各种YAML配置文件，包括RAG配置、向量数据库配置、提示词配置和智能体配置。
YAML格式示例：
    key: value
    nested:
      key: value
"""
import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    """
    加载RAG（检索增强生成）相关的配置文件

    Args:
        config_path: 配置文件路径，默认为config/rag.yml
        encoding: 文件编码，默认为utf-8

    Returns:
        dict: 包含RAG配置的字典对象
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    """
    加载Chroma向量数据库相关的配置文件

    Args:
        config_path: 配置文件路径，默认为config/chroma.yml
        encoding: 文件编码，默认为utf-8

    Returns:
        dict: 包含Chroma向量数据库配置的字典对象
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    """
    加载提示词配置文件，包含各类提示词的路径配置

    Args:
        config_path: 配置文件路径，默认为config/prompts.yml
        encoding: 文件编码，默认为utf-8

    Returns:
        dict: 包含提示词配置的字典对象
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    """
    加载智能体相关的配置文件

    Args:
        config_path: 配置文件路径，默认为config/agent.yml
        encoding: 文件编码，默认为utf-8

    Returns:
        dict: 包含智能体配置的字典对象
    """
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)



# 全局配置对象，在模块加载时自动初始化
# 这些配置对象在项目启动时加载一次，供整个项目使用
rag_conf = load_rag_config()        # RAG配置：包含聊天模型名称、向量数据库参数等
chroma_conf = load_chroma_config()  # Chroma向量数据库配置：包含数据库路径、集合名称等
prompts_conf = load_prompts_config()  # 提示词配置：包含各种提示词文件的路径
agent_conf = load_agent_config()    # 智能体配置：包含智能体的行为参数、工具配置等


if __name__ == '__main__':
    """
    测试代码：验证配置文件加载是否正常
    运行此文件时，会打印RAG配置中的chat_model_name字段
    """
    print(rag_conf["chat_model_name"])
