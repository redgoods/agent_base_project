"""
向量存储服务：负责将知识文档转换为向量并存入 Chroma 向量数据库
核心流程：读取文件 → 文本分片 → 向量化(embedding) → 存入向量库 → 提供检索接口
"""
from langchain_chroma import Chroma                      # Chroma 向量数据库的 LangChain 封装
from langchain_core.documents import Document              # LangChain 文档对象，携带文本内容和元数据
from utils.config_handler import chroma_conf               # Chroma 配置文件（路径、分片大小等参数）
from model.factory import embed_model                      # 预配置好的文本 embedding 模型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符级文本分片器
from utils.path_tool import get_abs_path                   # 获取绝对路径的工具函数
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex  # 文件读取和MD5计算
from utils.logger_handler import logger                    # 日志记录器
import os


class VectorStoreService:
    """
    向量存储服务类 —— 整个 RAG 系统的"记忆中枢"

    原理：
        大模型本身不知道你的私有知识（比如产品手册、公司文档）。
        这个类的作用就是把你的知识文档"消化"成向量存入数据库，
        以便后续用户提问时能快速找到相关段落。

    关键概念 —— Embedding（向量化）：
        把一段文字变成一个很长的数字数组（向量），语义相近的文字
        在向量空间中的距离也近。比如"猫"和"狗"的向量距离，
        比"猫"和"电脑"的向量距离更近。
        这样提问时，就能找到语义最相关的知识段落。
    """

    def __init__(self):
        # ========== 1. 初始化向量数据库 ==========
        # Chroma 是一个本地向量数据库，不需要额外安装数据库服务
        # 它会把向量数据持久化到磁盘（persist_directory），下次启动自动加载
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],   # 集合名称，类似数据库中的"表名"
            embedding_function=embed_model,                    # embedding 模型：负责把文字转成向量
            persist_directory=chroma_conf["persist_directory"], # 向量数据在磁盘上的存储路径
        )

        # ========== 2. 初始化文本分片器 ==========
        # 为什么需要分片？
        #   一篇 5000 字的文章直接转成一个向量，会丢失细节。
        #   切成 200 字一段，每段各自转成向量，检索时只召回最相关的几段。
        # 分片就像把一本书拆成段落卡片，每张卡片单独编目，方便查找。
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],           # 每段的最大字符数（如 500）
            chunk_overlap=chroma_conf["chunk_overlap"],      # 相邻段之间的重叠字符数（如 50），避免关键信息被切在两段中间而丢失上下文
            separators=chroma_conf["separators"],            # 分段优先级：先按段落分 → 再按句子 → 再按单词 → 最后按字符
            length_function=len,                             # 计算文本长度的函数
        )

    def get_retriever(self):
        """
        获取检索器 —— 用于根据用户问题查找相关文档

        原理：
            用户提问 "迷路怎么办" → 问题先转成向量 →
            在向量库中找距离最近的 k 个文档段落 → 返回这些段落
            这里的 k 就是配置中设定的返回数量（如返回最相关的 3 段）

        """
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹读取知识文件，转为向量存入向量库

        整体流程：
            ① 扫描数据文件夹，找出所有允许类型的文件（.txt / .pdf）
            ② 对每个文件计算 MD5，检查是否已处理过（去重）
            ③ 读取文件内容 → 拆分为 Document 对象列表
            ④ 用 TextSplitter 把长文档切成小段
            ⑤ 每一段通过 embedding 模型转成向量，存入 Chroma
            ⑥ 记录文件 MD5，下次不再重复处理

        为什么要用 MD5 去重？
            如果每次启动都重新加载所有文件，会重复存入相同的向量，
            浪费存储空间，检索时也会出现重复结果。
            MD5 就像文件的"指纹"，内容不变则 MD5 不变，
            通过比对指纹就知道文件是否已经处理过了。
        """

        def check_md5_hex(md5_for_check: str):
            """
            检查某个文件的 MD5 是否已经记录在案（即已存入向量库）
            读取 md5_hex_store 文件，逐行比对
            返回 True = 已存在，False = 未处理过
            """
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 如果记录文件不存在，先创建一个空文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # 找到了相同的 MD5，文件已处理过
                return False  # 遍历完没找到，这是新文件

        def save_md5_hex(md5_for_save: str):
            """
            将已处理文件的 MD5 追加记录到文件中
            这样下次启动时就知道这个文件已经加载过了
            """
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_document(read_path: str):
            """
            根据文件扩展名选择对应的读取方式
            .txt → 直接用 txt_loader 读取
            .pdf → 用 pdf_loader 解析 PDF 提取文字
            返回 list[Document]，每个 Document 包含 page_content（文本）和 metadata（元数据）
            """
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        # 扫描数据文件夹，获取所有允许加载的文件路径列表
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),            # 知识文件存放的目录
            tuple(chroma_conf["allow_knowledge_file_type"]),   # 允许的文件类型，如 ('.txt', '.pdf')
        )

        # 逐个处理每个文件
        for path in allowed_files_path:
            # ① 计算文件 MD5（文件的唯一指纹）
            md5_hex = get_file_md5_hex(path)

            # ② 检查是否已经处理过
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue  # 已存在，跳过这个文件

            try:
                # ③ 读取文件内容，转为 LangChain Document 对象
                documents: list[Document] = get_file_document(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # ④ 把长文档切成小段（分片），每段是一个独立的 Document
                # 例如一篇 3000 字的文章可能被切成 6 段，每段 500 字
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # ⑤ 将分片后的文档存入向量库
                # 内部流程：每段文字 → embedding 模型转成向量 → 存入 Chroma 数据库
                self.vector_store.add_documents(split_document)

                # ⑥ 记录文件 MD5，标记这个文件已经处理完毕
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # 某个文件加载失败不影响其他文件，记录错误后继续处理下一个
                # exc_info=True 会在日志中打印完整的错误堆栈，方便调试
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)
