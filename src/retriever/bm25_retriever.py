import os
import pickle
import jieba
from tqdm import tqdm
from typing import List, Any, Tuple
from .bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, base_dir="data/db/bm_corpus", db_name = "bm25_data") -> None:
        self.data_list = []
        self.tokenized_corpus = []
        self.base_dir = base_dir
        self.db_name = db_name

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
        
    def build(self, txt_list: List[str]):
        self.data_list = txt_list.copy()  # 防御性拷贝，防止外部列表修改影响内部状态
        
        for doc in tqdm(self.data_list, desc="bm25 build "):
            self.tokenized_corpus.append(self.tokenize(doc['content']))
        # 初始化 BM25Okapi 实例
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.save_bm25_data()
        
    def add(self, new_txt_list: List[str], auto_save: bool = False):
        """增量添加新文本并更新模型"""
        if not hasattr(self, 'bm25') or self.bm25 is None:
            raise ValueError("BM25模型未初始化，请先调用build或load_bm25_data")

        # 对新文本分词
        new_tokenized = [self.tokenize(doc['content']) for doc in tqdm(new_txt_list, desc="增量添加文本")]

        # 更新数据
        self.data_list.extend(new_txt_list)
        self.tokenized_corpus.extend(new_tokenized)

        # 重新构建BM25模型（BM25Okapi不支持增量更新，必须重新初始化）
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 可选：自动保存
        if auto_save:
            self.save_bm25_data()

    def tokenize(self,  text: str) -> List[str]:
        """ 使用jieba进行中文分词。
        """
        return list(jieba.cut_for_search(text))

    def save_bm25_data(self, db_name="bm25_data"):
        """ 对数据进行分词并保存到文件中。
        """
        db_file_path = os.path.join(self.base_dir, db_name + ".pkl")
        # 保存分词结果
        data_to_save = {
            "data_list": self.data_list,
            "tokenized_corpus": self.tokenized_corpus
        }
        with open(db_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_bm25_data(self, db_name= "bm25_data"):
        """ 从文件中读取分词后的语料库，并重新初始化 BM25Okapi 实例。
        """
        db_file_path = os.path.join(self.base_dir, db_name + ".pkl")
        
        if os.path.exists(db_file_path):
            with open(db_file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.data_list = data["data_list"]
            self.tokenized_corpus = data["tokenized_corpus"]
            
            # 重新初始化 BM25Okapi 实例
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            return True
        else:
            return False
    
    def search(self, query: str, top_n=5) -> List[Tuple[int, str, float]]:
        """ 使用BM25算法检索最相似的文本。
        """
        if self.tokenized_corpus is None:
            raise ValueError("Tokenized corpus is not loaded or generated.")

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取分数最高的前 N 个文本的索引
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        # 构建并返回结果列表
        result = [
            (i, self.data_list[i], scores[i])
            for i in top_n_indices
        ]

        return result