"""
Agent 工具集 —— 智能体的"手脚"

工具（Tool）是什么？
    工具就是普通的 Python 函数，加上 @tool 装饰器后，
    模型就能"看见"这些函数，知道它们叫什么、需要什么参数、返回什么。
    模型会根据用户问题自主决定调用哪个工具。

类比：
    模型是大脑，工具是手脚。大脑说"我需要查天气"，
    就会调用 get_weather 这个"手"去执行。

这里定义了两类工具：
    ① 基础查询工具：rag_summarize、get_weather、get_user_location 等
    ② 报告生成工具：fetch_external_data、fill_context_for_report
       用于生成用户使用报告的特殊场景
"""
import os
from langchain_core.tools import tool              # @tool 装饰器：把普通函数注册为 Agent 可用工具
from rag.rag_service import RagSummarizeService    # RAG 总结服务
import random                                       # 模拟数据用（实际项目中应替换为真实数据源）
from utils.config_handler import agent_conf         # Agent 配置文件
from utils.path_tool import get_abs_path            # 获取绝对路径
from utils.logger_handler import logger             # 日志记录器

# ========== 初始化 RAG 服务（全局单例） ==========
# 在模块加载时创建一次，避免每次调用工具都重新初始化
rag = RagSummarizeService()

# ========== 模拟数据池（实际项目中应从数据库获取） ==========
# 随机分配用户 ID
user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
# 随机分配月份
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

# 外部数据缓存 —— 从文件读取后存在内存中，避免重复读取
external_data = {}


# ========== 基础查询工具 ==========

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    """
    RAG 知识检索工具

    当模型需要查找私有知识时调用此工具。
    例如用户问"扫地机器人怎么清洁滤网"，模型会调用此工具
    去向量库中搜索相关产品手册内容。

    内部流程：
        问题 → 向量化 → 向量库检索 → 拼接参考资料 → 模型总结 → 返回回答

    参数 query: 检索关键词或问题
    返回: 基于参考资料生成的回答文本
    """
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """
    天气查询工具

    当用户问题涉及天气时，模型会调用此工具。
    例如"今天北京天气怎样，适合出门吗" → 模型调用 get_weather("北京")

    注意：当前返回的是模拟数据（固定晴天），实际项目中应接入真实天气 API。

    参数 city: 城市名称
    返回: 天气信息描述字符串
    """
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """
    获取用户位置工具

    当模型需要知道用户在哪里时调用。
    例如用户问"我这附近有没有门店"，模型会先调用此工具获取城市。

    注意：当前返回随机城市，实际项目中应从用户 session 或 GPS 获取。

    返回: 城市名称字符串
    """
    return random.choice(["北京", "上海", "广州", "深圳", "杭州", "西安", "武汉", "南京", "成都", "苏州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    """
    获取用户 ID 工具

    当模型需要用户身份时调用，如查询个人记录、生成个人报告等。

    注意：当前返回随机 ID，实际项目中应从用户登录 session 获取。

    返回: 用户 ID 字符串
    """
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    """
    获取当前月份工具

    当模型需要时间信息时调用，如查询某个月的使用记录。
    例如"我这个月用了多少次" → 模型先获取当前月份，再查记录。

    注意：当前返回随机月份，实际项目中应返回真实当前月份。

    返回: 月份字符串，格式如 "2025-06"
    """
    return random.choice(month_arr)


# ========== 报告生成相关工具 ==========

def generate_external_data():
    """
    从外部 CSV 文件加载用户使用数据，存入内存字典

    数据文件结构（CSV 格式，第一行为表头）：
        user_id, feature, efficiency, consumables, comparison, time
        1001, 小户型, 95%, 低, 优于平均, 2025-01
        1001, 大户型, 88%, 中, 持平, 2025-02
        ...

    转换为嵌套字典结构：
    {
        "1001": {
            "2025-01": {"特征": "小户型", "效率": "95%", "耗材": "低", "对比": "优于平均"},
            "2025-02": {"特征": "大户型", "效率": "88%", "耗材": "中", "对比": "持平"},
        },
        "1002": {
            ...
        },
    }

    懒加载设计：
        只在第一次调用时读取文件，之后直接从内存字典获取。
        避免每次查询都重新读文件。
    """
    # 如果字典已有数据，说明之前加载过了，直接返回
    if not external_data:
        external_data_path = get_abs_path(agent_conf['external_data_path'])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        # 逐行读取 CSV 文件（跳过表头行）
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")     # 用户 ID
                feature: str = arr[1].replace('"', "")      # 使用特征（如"小户型"）
                efficiency: str = arr[2].replace('"', "")   # 效率指标
                consumables: str = arr[3].replace('"', "")  # 耗材使用情况
                comparison: str = arr[4].replace('"', "")   # 与平均水平的对比
                time: str = arr[5].replace('"', "")         # 月份

                # 初始化用户条目
                if user_id not in external_data:
                    external_data[user_id] = {}

                # 存入对应月份的数据
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取用户在指定月份的使用记录, 以纯字符串形式返回, 如果未检索到就返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    外部数据查询工具

    当模型需要获取用户具体使用记录时调用。
    例如"我上个月用了多少次" → 模型拿到 user_id 和 month 后调用此工具。

    内部流程：
        ① 确保数据已加载（懒加载）
        ② 按 user_id + month 在嵌套字典中查找
        ③ 找到返回数据字典，找不到返回空字符串

    参数 user_id: 用户 ID，如 "1001"
    参数 month: 月份，如 "2025-01"
    返回: 该月份的使用记录字典的字符串表示，或空字符串
    """
    # 确保数据已加载到内存
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        # 用户 ID 或月份不存在，记录警告日志
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    """
    报告场景上下文注入工具 —— 这是一个"信号工具"

    它本身不做任何实质性工作，返回的字符串也没有实际用途。
    它真正的价值在于：中间件 monitor_tool 会监测到这个工具被调用，
    然后自动把 runtime.context["report"] 设为 True。

    这触发了后续的提示词切换：
        report = True → resport_prompt_switch 返回报告专用提示词
        模型的"人设"从普通客服切换为报告生成器

    类比：
        就像按了一个"切换到报告模式"的按钮。
        按钮本身没什么功能，但按下后系统进入另一种工作状态。

    无入参，无实际返回值。
    """
    return "fill_context_for_report已调用"







