# 智扫通 — 扫地机器人 AI 智能客服

基于 **LangChain + LangGraph + RAG + ReAct Agent** 的扫地机器人智能客服系统，具备私有知识检索、天气适配、个人使用报告生成等能力。

---

## 一、项目简介

这是一个面向扫地/扫拖一体机器人场景的 **AI 智能客服**，相比传统固定流程的问答系统，它能自主思考需要什么信息、调用哪些工具、如何组合答案。

### 核心能力

| 能力 | 说明 |
|---|---|
| 📚 私有知识问答 | 基于产品手册、使用指南等私有文档回答专业问题 |
| 🌤️ 环境适配建议 | 结合天气数据判断当前环境是否适合使用机器人 |
| 📊 个人使用报告 | 自动生成用户月度使用报告（清洁效率、耗材状态等） |
| 🔀 场景自动切换 | 同一客服自动识别意图，在普通问答和报告生成间切换人设 |

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────┐
│  app.py           ← Streamlit Web 聊天界面           │
├─────────────────────────────────────────────────────┤
│  react_agent.py   ← ReAct Agent（智能体大脑）        │
│                     自主推理 → 选择工具 → 综合回答    │
├──────────────────┬──────────────────────────────────┤
│  agent_tools.py  │  middleware.py                    │
│  工具（手脚）     │  中间件（监控器）                  │
│  · rag_summarize │  · monitor_tool                   │
│  · get_weather   │  · log_before_model               │
│  · get_user_loc  │  · resport_prompt_switch          │
│  · get_user_id   │                                   │
│  · get_month     │                                   │
│  · fetch_data    │                                   │
│  · fill_context  │                                   │
├─────────────────────────────────────────────────────┤
│  rag/                                             │
│  · vector_store.py   ← Chroma 向量数据库            │
│  · rag_service.py    ← RAG 检索 + 总结服务          │
└─────────────────────────────────────────────────────┘
```

---

## 三、项目结构

```
agent 项目/
├── app.py                      # Streamlit 前端界面
├── agent/
│   ├── react_agent.py          # ReAct Agent 核心控制器
│   └── tools/
│       ├── agent_tools.py      # 7 个工具函数定义
│       └── middleware.py       # 3 个中间件（监控器）
├── rag/
│   ├── vector_store.py         # Chroma 向量存储服务
│   └── rag_service.py          # RAG 检索 + 总结服务
├── model/
│   └── factory.py              # 模型工厂（LLM + Embedding）
├── utils/
│   ├── config_handler.py       # YAML 配置读取
│   ├── path_tool.py            # 路径工具
│   ├── file_handler.py         # PDF/TXT 文件读取
│   ├── prompt_loader.py        # 提示词加载
│   └── logger_handler.py       # 日志处理
├── config/
│   ├── chroma.yml              # 向量库配置（分片大小、集合名等）
│   ├── agent.yml               # Agent 配置（外部数据路径）
│   └── rag.yml                 # 模型配置（模型名称）
├── prompts/
│   ├── main_prompt.txt         # 系统主提示词（Agent 人设）
│   ├── rag_summarize.txt       # RAG 总结提示词
│   └── report_prompt.txt       # 报告生成提示词
├── data/                       # 知识库文件（PDF/TXT）
├── data/external/
│   └── records.csv             # 用户使用记录数据
├── logs/                       # 运行日志
└── chroma_db/                  # 向量数据库持久化存储
```

---

## 四、快速开始

### 环境要求

- Python 3.10+

### 安装依赖

```bash
pip install streamlit langchain langgraph langchain-chroma
```

### 配置

修改以下配置文件：

| 文件 | 关键配置项 |
|---|---|
| `config/rag.yml` | `chat_model_name`（大模型名称）、`embedding_model_name`（嵌入模型名称） |
| `config/chroma.yml` | `chunk_size`（分片大小）、`k`（检索返回数量） |

### 运行

#### 1. 加载知识库到向量库

```bash
python -m rag.vector_store
```

首次运行会自动扫描 `data/` 目录下的 `.txt` / `.pdf` 文件，分片后转为向量存入 Chroma。

#### 2. 启动 Web 界面

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可开始对话。

---

## 五、核心原理

### 5.1 RAG（检索增强生成）

大模型不知道你的私有知识，RAG 的做法是：

```
用户提问 → 问题向量化 → 向量库检索相关段落 →
拼接成上下文 → 连同问题一起交给模型 → 模型"看着资料"回答
```

**为什么用语义搜索而不是关键词搜索？**

用户问"滤网清洗方法"，文档写的是"清洁过滤网"，关键词对不上。
Embedding 模型会把语义相近的文字映射到向量空间中相近的位置，
即使字面不完全一致也能搜到相关内容。

### 5.2 ReAct Agent（推理 + 行动）

相比固定流程的 RAG，Agent 能自主决策：

```
普通 RAG：  问题 → 检索 → 回答（固定流程）
Agent：    问题 → 模型思考需要什么 → 自主选择工具 → 拿到结果 → 可能需要更多工具 → 最终回答
```

### 5.3 中间件机制

中间件在 Agent 运行时自动触发，不直接参与问答，而是旁路监控：

| 中间件 | 触发时机 | 作用 |
|---|---|---|
| `monitor_tool` | 每次工具调用 | 记录日志 + 识别报告场景 |
| `log_before_model` | 每次模型调用前 | 打印对话状态（调试用） |
| `resport_prompt_switch` | 每次模型调用前 | 根据场景动态切换提示词 |

### 5.4 报告生成流程

同一个 Agent 通过中间件自动切换人设：

```
用户："给我生成月度使用报告"
  → 模型调用 fill_context_for_report（信号工具）
  → monitor_tool 捕获，设置 context["report"] = True
  → resport_prompt_switch 检测到 report=True
  → 切换为报告专用提示词（人设变为报告生成器）
  → 获取用户 ID → 获取月份 → 查询使用记录 → 生成报告
```

---

## 六、工具一览

| 工具 | 功能 | 入参 |
|---|---|---|
| `rag_summarize` | RAG 知识检索 | `query`（检索词） |
| `get_weather` | 查询城市天气 | `city`（城市名） |
| `get_user_location` | 获取用户所在城市 | 无 |
| `get_user_id` | 获取用户 ID | 无 |
| `get_current_month` | 获取当前月份 | 无 |
| `fetch_external_data` | 查询用户使用记录 | `user_id`, `month` |
| `fill_context_for_report` | 触发报告场景切换 | 无 |

---

## 七、已知问题

- `middleware.py` 中 `monitor_tool` 设置的上下文键名为 `"report"`，但 `resport_prompt_switch` 读取的键名为 `"resport"`（拼写不一致），导致报告场景的提示词切换可能不会生效，需统一拼写。

---

## 八、配置说明

### chroma.yml

```yaml
collection_name: agent              # 向量库集合名称
persist_directory: chroma_db        # 向量数据持久化路径
k: 3                                # 每次检索返回的相关段落数量
data_path: data                     # 知识库文件目录
md5_hex_store: md5.text             # 已处理文件 MD5 记录文件
allow_knowledge_file_type: ["txt", "pdf"]  # 允许加载的文件类型

chunk_size: 200                     # 文本分片大小（每段字符数）
chunk_overlap: 20                   # 相邻分片重叠字符数（防止信息被切断）
separators: ["\n\n", "\n", ".", ...]       # 分片分隔符优先级
```

### rag.yml

```yaml
chat_model_name: qwen3-max          # 对话模型
embedding_model_name: text-embedding-v4  # 文本嵌入模型
```
