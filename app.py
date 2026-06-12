"""
Streamlit Web应用 —— "智扫通"智能客服的前端界面

Streamlit 是一个纯 Python 的 Web 框架，不需要写 HTML/CSS/JS，
用 Python 代码就能快速搭建交互式 Web 应用。

整体架构：
    用户浏览器 ← Streamlit 界面 ← ReactAgent 智能体 ← 各工具 + RAG 服务
"""
import time                                 # 用于流式输出的微小延迟
import streamlit as st                      # Streamlit Web 框架
from agent.react_agent import ReactAgent    # ReAct 智能体核心类

# ========== 页面标题 ==========
st.title("智扫通机器人智能客服")
st.divider()  # 在标题下方画一条分隔线

# ========== 初始化 Session State ==========
# session_state 类似 Web 的"会话变量"，用于在多次页面刷新之间保持数据
# 页面每次交互都会重新运行整个脚本，所以需要检查是否已初始化

# ① 智能体实例 —— 只在首次加载时创建
#    后续刷新页面复用同一个 agent，避免重复初始化
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

# ② 消息历史 —— 存储所有对话记录
#    格式：[{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，有什么可以帮您？"}, ...]
if "message" not in st.session_state:
    st.session_state["message"] = []

# ========== 渲染历史消息 ==========
# 遍历 session_state 中存储的消息列表，逐条显示在界面上
# 这保证了每次页面刷新后聊天记录不会消失
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
    # role: "user" 显示在右侧，"assistant" 显示在左侧
    # write 把消息内容渲染到页面上

# ========== 用户输入 ==========
# 在页面底部显示一个聊天输入框，用户在这里输入问题
# prompt 是用户输入的文本，没有输入时为 None
prompt = st.chat_input()

# ========== 处理用户输入 ==========
if prompt:
    # ① 立即显示用户发送的消息
    st.chat_message("user").write(prompt)

    # ② 将用户消息追加到历史记录
    st.session_state["message"].append({"role": "user", "content": prompt})

    # ③ 准备接收模型回复的缓存列表
    response_messages = []

    # ④ 显示"思考中..."的加载提示
    with st.spinner("智能客服思考中..."):
        # ⑤ 调用 Agent 的流式执行方法，获取生成器
        #    res_stream 是一个迭代器，每次 yield 一小段生成的文本
        res_stream = st.session_state["agent"].execute_stream(prompt)

        # ⑥ 定义 capture 生成器函数
        #    作用：遍历模型的流式输出，一边输出给用户，一边缓存到 response_messages
        #    这解决了 Streamlit 的 write_stream 需要生成器，但我们也要保存完整回复的问题
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)  # 缓存这段文本

                # 逐字符 yield 出去，实现打字机效果
                for chat in chunk:
                    time.sleep(0.01)  # 微小的延迟，让打字机效果更自然
                    yield chat

        # ⑦ 流式显示 AI 回复
        #    write_stream 接收生成器，逐段显示，模拟"打字中"的效果
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))

        # ⑧ 将 AI 回复追加到历史记录
        #    response_messages[-1] 是最后一段缓存的文本（完整回复）
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})

        # ⑨ 重新渲染页面，刷新显示最新状态
        #    rerun 会让整个脚本重新运行，重新执行时会重新渲染所有历史消息
        st.rerun()
