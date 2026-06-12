"""
RAG 总结服务：用户提问 → 从向量库搜索参考资料 → 将问题和资料一起提交给模型 → 模型总结后回复

这就是整个 RAG（检索增强生成）系统的"问答引擎"。
它本身不存储知识，而是依赖 VectorStoreService 来检索知识，
然后把检索到的知识和用户问题打包，交给大模型生成回答。
"""

from langchain_core.documents import Document        # LangChain 文档对象，代表一段检索到的文本
from langchain_core.output_parsers import StrOutputParser  # 把模型输出解析为纯字符串
from rag.vector_store import VectorStoreService      # 向量存储服务，负责知识检索
from utils.prompt_loader import load_rag_prompts     # 加载 RAG 提示词模板
from langchain_core.prompts import PromptTemplate    # LangChain 提示词模板类
from model.factory import chat_model                 # 预配置好的大语言模型实例


def print_prompt(prompt):
    """
    调试辅助函数：在链式调用中打印当前构建好的提示词

    它在 LCEL 链中充当"透明管道"——打印内容后原样返回，
    不影响数据流向下一个环节。方便开发时观察提示词实际长什么样。
    """
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    """
    RAG 总结服务类 —— 整个 RAG 系统的"问答引擎"

    核心思路（为什么需要 RAG？）：
        大模型训练数据有截止日期，也不知道你的私有知识。
        直接问它"我们公司产品的保修政策是什么"，它答不上来。
        RAG 的做法是：先去你的知识库中查找相关资料，
        把查到的资料连同问题一起交给模型，让它"看着资料回答"。

        就像开卷考试——先翻书找到相关内容，再基于这些内容组织答案。

    工作流程（一次完整的问答）：
        ① 用户提问："小户型适合哪些扫地机器人？"
        ② 把问题转为向量，在向量库中找最相关的几段资料
        ③ 把这些资料拼接成一段"上下文"
        ④ 把"问题 + 上下文"填入提示词模板
        ⑤ 提示词送给大模型，模型阅读上下文后生成回答
        ⑥ 回答返回给用户
    """

    def __init__(self):
        # ① 创建向量存储服务，获取检索器
        #    retriever 就是"查资料的工具"——给它一个问题，它返回最相关的文档段落
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        # ② 加载提示词模板
        #    提示词决定了"怎么让模型看资料、怎么回答"
        #    例如："请根据以下参考资料回答问题：\n{context}\n\n问题：{input}\n回答："
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

        # ③ 获取大语言模型实例
        self.model = chat_model

        # ④ 组装完整的处理链（chain）
        #    这是 LangChain 的 LCEL（LangChain Expression Language）语法，
        #    用 | 符号把各个环节串起来，像流水线一样：
        #    填入模板 → 打印调试 → 送模型 → 解析输出
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化 LCEL 处理链 —— 一条从输入到输出的"流水线"

        链式结构（数据依次流过每个环节）：
            PromptTemplate → print_prompt → chat_model → StrOutputParser
            （填模板）        （调试打印）     （模型生成）    （转字符串）

        其中 PromptTemplate 负责把 {input} 和 {context} 两个变量
        填入模板文本，生成完整的提示词，交给模型。

        类比：就像工厂的装配线，每个工位做一件事，产品依次流转。
        """
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        检索相关文档 —— 去向量库中"查资料"

        原理：
            用户的提问先被 embedding 模型转成向量，
            然后在向量库中计算与所有文档段落的相似度，
            取最接近的 k 个段落（k 在配置中设定）。

            就像在图书馆按关键词搜书——只不过这里不是匹配文字，
            而是匹配语义。问"怎么清洁地板"能搜到"地面打扫技巧"，
            即使字面上不完全一致。

        参数 query: 用户的问题
        返回: 最相关的 k 个 Document 对象列表
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        RAG 总结 —— 一次完整的问答

        步骤拆解：
            步骤 ① 查资料：
                调用 retriever_docs，根据用户问题在向量库中找到相关段落
                例如问"小户型扫地机器人"，可能返回 3 段相关产品介绍

            步骤 ② 拼上下文：
                把找到的段落拼接成一段文字，每段加上编号和格式
                格式如：
                    【参考资料1】: 参考资料: 某产品适合小户型... | 参考元数据: {...}
                    【参考资料2】: 参考资料: 另一款产品也适合... | 参考元数据: {...}

                这一步就是给模型"翻书"——把查到的资料整理好给它看

            步骤 ③ 让模型回答：
                把用户问题（input）和整理好的资料（context）
                填入提示词模板，走处理链得到模型回复

        参数 query: 用户的问题
        返回: 模型基于参考资料生成的回答（字符串）
        """

        # ① 查资料：在向量库中检索与问题最相关的文档段落
        context_docs = self.retriever_docs(query)

        # ② 拼上下文：把检索到的段落拼接成格式化的文本
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            # 每段加上编号，方便模型知道这是第几条参考资料
            # doc.page_content 是段落文字，doc.metadata 是元数据（如来源文件名）
            context += f"【参考资料{counter}】: 参考资料: {doc.page_content} | 参考元数据: {doc.metadata}\n"

        # ③ 调用处理链：把问题和上下文填入提示词，让模型生成回答
        return self.chain.invoke(
            {
                "input": query,        # 用户的问题
                "context": context     # 检索到的参考资料（拼接后的文本）
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
