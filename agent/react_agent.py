"""
ReAct Agent —— 智能体的核心控制器

ReAct = Reasoning + Acting（推理 + 行动）
原理：模型不只是直接回答，而是先"思考"需要什么信息，
      然后决定调用哪些工具来获取信息，最后综合信息给出回答。

类比：就像一个客服接待客户：
  ① 听客户说什么（接收问题）
  ② 想想要查什么资料（推理）
  ③ 去系统里查数据（调用工具）
  ④ 综合所有信息回答客户（生成回复）
"""
from langchain.agents import create_agent            # LangChain 创建 Agent 的工厂函数
from model.factory import chat_model                  # 预配置的大语言模型（大脑）
from utils.prompt_loader import load_system_prompts   # 加载系统提示词（角色设定和行为指南）

# 导入所有可用工具 —— 这些是 Agent 可以"使用的手"
from agent.tools.agent_tools import (
    rag_summarize,         # RAG 知识检索：从向量库找参考资料
    get_weather,           # 天气查询
    get_user_location,     # 获取用户所在城市
    get_user_id,           # 获取用户 ID
    get_current_month,     # 获取当前月份
    fetch_external_data,   # 从外部系统获取用户使用记录
    fill_context_for_report,  # 触发报告生成场景的上下文注入
)

# 导入中间件 —— 这些是 Agent 运行时的"监控器"
from agent.tools.middleware import (
    monitor_tool,           # 工具调用监控：记录每次工具调用
    log_before_model,       # 模型调用前日志：记录模型收到的消息
    report_prompt_switch,  # 提示词动态切换：根据场景切换不同的提示词
)


class ReactAgent:
    """
    ReAct 智能体类 —— 整个系统的"大脑 + 手脚"

    工作流程：
        用户提问 → Agent 分析意图 → 决定调用哪些工具 →
        工具返回结果 → Agent 综合信息 → 生成回答 → 流式返回

    关键概念 —— Agent vs 普通问答：
        普通问答（RAG）：问题 → 检索 → 模型回答（固定流程）
        Agent 智能体：  问题 → 模型思考 → 自主选择工具 → 拿到结果 →
                       可能需要更多工具 → 最终回答（灵活决策）

        Agent 更聪明 —— 它自己判断需要什么信息，而不是固定流程。
    """

    def __init__(self):
        # 创建 Agent 实例，配置三个核心要素：
        self.agent = create_agent(
            # ① 模型（大脑）：负责推理和决策
            model=chat_model,

            # ② 系统提示词（人设和行为指南）：
            #    告诉模型"你是谁"、"你能做什么"、"该怎么回答"
            system_prompt=load_system_prompts(),

            # ③ 工具列表（手脚）：模型可以自主调用的函数
            #    模型会根据问题自动决定用哪些工具
            #    例如用户问"今天天气怎样" → 模型自动选择 get_weather
            tools=[
                rag_summarize,         # RAG 知识检索
                get_weather,           # 查询天气
                get_user_location,     # 获取用户位置
                get_current_month,     # 获取当前月份
                get_user_id,           # 获取用户 ID
                fetch_external_data,   # 获取外部使用记录
            ],

            # ④ 中间件（监控器）：在工具调用和模型调用前后执行额外逻辑
            middleware=[
                monitor_tool,           # 每次工具调用前后：记录日志
                log_before_model,       # 每次模型调用前：记录当前对话状态
                report_prompt_switch,  # 每次模型推理前：根据场景动态切换提示词
            ],
        )

    def execute_stream(self, query: str):
        """
        流式执行 Agent 推理过程，逐块返回结果

        流式输出的意义：
            不用等模型完全生成完才显示，而是一边生成一边展示，
            用户体验更好——就像 ChatGPT 那样一个字一个字蹦出来。

        stream_mode="values" 的含义：
            每次返回 Agent 状态的完整快照（所有累积的消息），
            我们取最后一条消息就是最新生成的内容。

        参数 query: 用户的问题
        返回: 生成器，每次 yield 一段文本内容
        """
        # 构建输入消息格式 —— LangGraph 要求的消息结构
        input_dict = {
            "messages": [
                {"role": "user", "content": query},  # 用户消息
            ]
        }

        # stream 返回一个迭代器，每次迭代是 Agent 运行中的一个状态快照
        # context={"report": False} 初始上下文，标记当前不是报告生成场景
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            # 取出最新的消息（就是这一步新生成的内容）
            latest_message = chunk["messages"][-1]

            # 如果有实际内容就 yield 出去，供前端流式显示
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
