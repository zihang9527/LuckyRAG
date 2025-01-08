import sys
from logger import logger


if __name__ == '__main__':
    # import os.path
    # 使用示例
    # file_path = "path/to/your/file.txt"  # 替换为你要获取上级目录的文件路径
    # parent_dir = os.path.dirname(file_path)
    # print(f"文件 {file_path} 的上级目录是: {parent_dir}")
    # print(parent_dir+'/1.txt')

    from retriever.bm25_retriever import BM25Retriever
    corpus = [
        "Hello there good man!",
        "It is quite windy in London",
        "How is the weather today?"
    ]
    bm_retriever = BM25Retriever(txt_list=corpus)
    query = "windy London"
    results = bm_retriever.search(query, top_n=3)

    from reranker.reranker_bge_m3 import RerankerBGEM3

    reranker = RerankerBGEM3(model_id_key="BAAI/bge-m3-base-en-v1.5", device = 'cpu', is_api=False)

    reranker_results = reranker.rank(query, [result[1] for result in results])
    print(reranker_results)
    


    