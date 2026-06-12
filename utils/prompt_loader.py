"""
提示词加载模块
负责从配置文件和文本文件中加载各类提示词（prompt）

提示词是LLM（大语言模型）的重要输入，用于引导模型生成预期的输出。
本模块提供统一的提示词加载接口，支持系统提示词、RAG提示词、报告生成提示词等。

主要功能：
- 从YAML配置中读取提示词文件路径
- 从文本文件中加载提示词内容
- 错误处理和日志记录
"""
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts():
    """
    加载系统提示词（System Prompt）

    系统提示词定义了AI助手的基本角色和行为准则，
    例如："你是一个专业的AI助手，擅长回答问题..."

    Returns:
        str: 系统提示词的文本内容

    Raises:
        KeyError: 如果配置文件中缺少"main_prompt_path"配置项
        FileNotFoundError: 如果提示词文件不存在
        Exception: 其他读取错误
    """
    try:
        # 从配置中获取系统提示词文件的路径
        # prompts_conf是从prompts.yml加载的配置字典
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])

    except KeyError as e:
        # 配置项不存在时的错误处理
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        # 读取提示词文件内容
        # open()打开文件，read()读取全部内容
        return open(system_prompt_path, "r", encoding="utf-8").read()

    except Exception as e:
        # 文件读取错误处理
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_rag_prompts():
    """
    加载RAG总结提示词（RAG Summarization Prompt）

    RAG总结提示词用于在检索增强生成（RAG）系统中，
    对检索到的文档内容进行总结和提炼，为用户提供简洁准确的回答。

    Returns:
        str: RAG总结提示词的文本内容

    """
    try:
        # 从配置中获取RAG总结提示词文件的路径
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])

    except KeyError as e:
        # 配置项不存在时的错误处理
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        # 读取提示词文件内容
        return open(rag_prompt_path, "r", encoding="utf-8").read()

    except Exception as e:
        # 文件读取错误处理
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e

def load_report_prompts():
    """
    加载报告生成提示词（Report Generation Prompt）

    报告生成提示词用于指导LLM生成结构化的报告文档，
    例如分析报告、总结报告、技术文档等。

    Returns:
        str: 报告生成提示词的文本内容

    """
    try:
        # 从配置中获取报告生成提示词文件的路径
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])

    except KeyError as e:
        # 配置项不存在时的错误处理
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        # 读取提示词文件内容
        return open(report_prompt_path, "r", encoding="utf-8").read()

    except Exception as e:
        # 文件读取错误处理
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e

if __name__ == '__main__':
    """
    测试代码：验证提示词加载功能

    运行此文件时，会测试报告生成提示词的加载功能
    """
    try:
        # 测试加载报告生成提示词
        report_prompt = load_report_prompts()
        print("=" * 50)
        print("报告生成提示词加载成功：")
        print("=" * 50)
        print(report_prompt)
        print("=" * 50)

    except Exception as e:
        print(f"测试失败：{str(e)}")

