"""
Agent 中间件 —— 智能体运行时的"监控器"和"控制器"

中间件（Middleware）是什么？
    中间件是在 Agent 运行过程中，在特定时机自动执行的额外逻辑。
    它不直接参与问答，而是"旁路监控"——在工具调用前后、模型调用前后
    等关键时刻插入自定义行为。

类比：
    如果 Agent 是生产线，工具是工人，模型是质检员，
    那中间件就是车间主任——不亲自干活，但监督每个环节，
    必要时调整工作流程。

本文件定义了三个中间件：
    ① monitor_tool     —— 工具调用监控器（记录日志 + 场景标记）
    ② log_before_model  —— 模型调用前日志（记录对话状态）
    ③ resport_prompt_switch —— 提示词动态切换（根据场景换提示词）
"""
from typing import Callable                                    # 类型提示：可调用的函数类型
from utils.prompt_loader import load_system_prompts, load_report_prompts  # 加载不同场景的提示词
from langchain.agents import AgentState                        # Agent 的状态对象，包含所有对话消息
from langchain.agents.middleware import (                      # LangChain 中间件装饰器
    wrap_tool_call,       # @wrap_tool_call: 包裹工具调用，在工具执行前后插入逻辑
    before_model,         # @before_model: 在模型调用前执行
    dynamic_prompt,       # @dynamic_prompt: 动态决定使用哪个提示词
    ModelRequest,         # 模型请求对象，包含当前状态和上下文
)
from langchain.tools.tool_node import ToolCallRequest          # 工具调用请求对象
from langchain_core.messages import ToolMessage                # 工具返回消息
from langgraph.runtime import Runtime                          # 运行时上下文，携带执行过程中的变量
from langgraph.types import Command                            # 控制流命令类型
from utils.logger_handler import logger                        # 日志记录器


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装 —— 包含工具名、参数等调用信息
        request: ToolCallRequest,
        # 实际要执行的工具函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    """
    工具调用监控中间件

    每次 Agent 调用任何工具时，这个函数都会先执行：
        ① 记录日志（哪个工具、什么参数）
        ② 执行真正的工具函数
        ③ 记录结果（成功/失败）
        ④ 特殊处理：如果是 fill_context_for_report，标记进入报告场景

    这就是"旁路监控"——不改变工具的实际行为，只是多了一层观察和控制。

    @wrap_tool_call 装饰器的作用：
        LangChain 会在每次工具调用时自动调用这个函数，
        由这个函数决定要不要执行真正的工具（handler），以及怎么处理结果。

    参数 request: 工具调信息，包含 tool_call.name（工具名）和 tool_call.args（参数）
    参数 handler: 实际的工具函数，调用 handler(request) 才会真正执行工具
    返回: 工具执行结果（ToolMessage）或控制流命令（Command）
    """
    # ① 记录调用日志
    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数: {request.tool_call['args']}")

    try:
        # ② 执行真正的工具函数
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        # ③ 特殊场景处理：报告生成上下文标记
        # 当模型调用 fill_context_for_report 工具时，说明用户想生成报告，
        # 把 runtime.context["report"] 设为 True，通知其他中间件切换工作模式
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        # 工具调用失败时记录错误日志
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e  # 重新抛出异常，让上层知道工具调用失败了


@before_model
def log_before_model(
        state: AgentState,          # 整个 Agent 智能体中的状态记录，包含所有历史消息
        runtime: Runtime,           # 运行时上下文，记录了执行过程中的各种变量
):
    """
    模型调用前日志中间件

    每次 Agent 准备让模型生成回复之前，这个函数都会先执行。
    它的作用是记录当前 Agent 的对话状态，方便调试和排查问题。

    @before_model 装饰器的作用：
        在模型开始生成回复前自动执行此函数。
        如果返回 None（本函数就是），模型正常执行。
        如果返回其他值，可能会干预模型的输入。

    参数 state: Agent 的当前状态，最重要的 state['messages'] 是所有对话消息列表
    参数 runtime: 运行时上下文，包含自定义变量（如上面的 context["report"]）
    返回: None（不干预模型执行）
    """
    # 记录即将调用模型，以及当前有多少条对话消息
    logger.info(f"[log_before_model]即将调用模型, 带有{len(state['messages'])}条消息。")

    # DEBUG 级别日志：打印最后一条消息的类型和内容
    # 这能看到模型最后收到的是什么（用户问题？工具返回？系统提示？）
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    """
    提示词动态切换中间件

    核心功能：根据当前场景，动态决定使用哪套提示词。

    为什么要动态切换提示词？
        同一个 Agent 在不同场景下需要不同的"人设"：
        - 普通问答：你是客服，回答用户问题
        - 报告生成：你是分析师，基于用户数据生成报告

        用同一个 Agent，切换提示词就能扮演不同角色。

    @dynamic_prompt 装饰器的作用：
        每次模型要生成回复前，自动调用此函数获取应该使用的提示词。
        返回什么提示词，模型就按什么指示工作。

    工作流程：
        用户说"给我生成使用报告"
        → 模型决定调用 fill_context_for_report 工具
        → monitor_tool 捕获到调用，设置 context["report"] = True
        → 下次模型调用前，resport_prompt_switch 检查 context
        → 发现 report = True → 返回报告专用提示词
        → 模型按报告提示词的风格生成回复

    参数 request: 模型请求，包含运行时上下文（context 中的变量）
    返回: 提示词文本 —— 报告场景返回报告提示词，否则返回默认系统提示词
    """
    # 检查当前是否处于报告生成场景
    is_report = request.runtime.context.get("report", False)

    if is_report:
        # 报告场景：返回报告生成专用提示词
        # 报告提示词可能包含"请根据以下数据为用户生成月度使用报告"之类的指示
        return load_report_prompts()

    # 普通场景：返回默认系统提示词
    # 系统提示词定义了 Agent 的一般行为和可用工具说明
    return load_system_prompts()


