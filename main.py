# import os
# import json
# from typing import Dict, List, Optional, Tuple, Union

# from embedding.qwen_embedding import QwenEmbedding
# from llm.qwen_llm import QwenLLM
# from parser.txt_parser import TXTParser

# import chromadb

# api_key = ''
# embedding = QwenEmbedding(api_key=api_key)

# file_path = 'data/data.txt'
# parser = TXTParser(file_path=file_path, model=embedding)
# parser.parse()

# # print(parser.parse_output[0])
# # print(len(parser.parse_output))

# client = chromadb.Client()
# test_collection = client.create_collection(name="test")

# for id, dic in enumerate(parser.parse_output):
#     test_collection.add(
#         documents=[dic['content']],
#         embeddings=[dic['embedding']],
#         ids= str(id)
#     )

# query = 'what is the Qwen2.5-math model?'
# qwen = QwenLLM(model_id_key=api_key, is_api=True)
# qwen_result = qwen.generate(query)
# print(qwen_result)

# embeddings_query = embedding.get_embedding(query)

# db_results = test_collection.query(
#     query_embeddings=[embeddings_query],
#     n_results=2
# )
# temp = []
# for ls in db_results['documents']:
#     for s in ls:
#         temp.append(s)
# context = "\n".join(temp)

# RAG_PROMPT_TEMPALTE="""参考信息：
# {context}
# ---
# 我的问题或指令：
# {question}
# ---
# 我的回答：
# {answer}
# ---
# 请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
# 你修正的回答:"""

# prompt_text = RAG_PROMPT_TEMPALTE.format(
#         context=context,
#         question=query,
#         answer=qwen_result
#     )
# qwen_result = qwen.generate(prompt_text)
# print(qwen_result)

# from typing import Collection
# import embedding
# from embedding.bge_embedding import BgeEmbedding
# from llm.qwen_llm import QwenLLM
# from parser.txt_parser import TXTParser
# import chromadb

# RAG_PROMPT_TEMPALTE="""参考信息：
# {context}
# ---
# 我的问题或指令：
# {question}
# ---
# 请根据上述参考信息回答和我的问题或指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来准确回答我的问题。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
# 你的回答:"""

# embedding = BgeEmbedding(model_path='', is_api=False)
# qwen = QwenLLM(model_id_key='', is_api=False)

# # docs embedding
# file_path = 'data/data.txt'
# parser = TXTParser(file_path=file_path, model=embedding)
# parser.parse()
# # # print(parser.parse_output[0])
# # # print(len(parser.parse_output))

# # db
# client = chromadb.Client()
# test_collection = client.create_collection(name="test")
# for id, dic in enumerate(parser.parse_output):
#     test_collection.add(
#         documents=[dic['content']],
#         embeddings=[dic['embedding']],
#         ids= str(id)
#     )

# # rag
# query = 'what is the Qwen2.5-math model?'
# embeddings_query = embedding.get_embedding(query)
# db_results = test_collection.query(
#     query_embeddings=[embeddings_query],
#     n_results=2
# )

# temp = []
# for ls in db_results['documents']:
#     for s in ls:
#         temp.append(s)
# context = "\n".join(temp)

# prompt_text = RAG_PROMPT_TEMPALTE.format(
#         context=context,
#         question=query,
#     )
# qwen_result = qwen.generate(prompt_text)
# print(qwen_result)

from typing import Collection
import math
import random
from tqdm import tqdm
import os
import json
import datetime
import time

from argument import Argument

from logger import logger
from src.embedding.bge_embedding import BgeEmbedding
from src.llm.qwen_llm import QwenLLM
from src.retriever.chroma_retriever import ChromaRetriever
from src.parser.pdf_parser import PdfParser

def main():
    pdf_parser = PdfParser(ocr_model_path='./data/ocr_model', artifacts_path='./data/artifacts')
    parse_result = pdf_parser.parse(file_path_or_url='./data/test.pdf')
    file_name = pdf_parser.get_file_name(parse_result)
    md_result = pdf_parser.convert_to_file(parse_result, ouput_type='md')

    from src.splitter.md_spliter import MDSpliter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_spliter = MDSpliter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = md_spliter.split(markdown_document = md_result)

    from src.splitter.recursive_char_spliter import RecursiveTextSplitter
    chunk_size = 250
    chunk_overlap = 15
    r_spliter = RecursiveTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    splits = r_spliter.split(document = splits)
    
    from src.embedding.bge_embedding import BgeEmbedding
    embedding_model = ''
    embedding = BgeEmbedding(model_path = embedding_model, is_api=False)

    db_dataset = []
    for split in splits:
        data_item = {}

        metadata = {}
        metadata['filename'] = file_name
        metadata['position'] = split.metadata
        
        data_item['content'] = split.page_content
        data_item['metadata'] =  metadata
        data_item['embedding'] = embedding.get_embedding(data_item['content'])

        db_dataset.append(data_item)
    
    print(len(db_dataset))

    from src.retriever.chroma_retriever import ChromaRetriever
    persist_file_path = ''
    collection_name = 'test'

    db = ChromaRetriever(persist_file_path=persist_file_path, embedding=embedding)

    if db.load(collection_name=collection_name) == False:
        db.create(collection_name=collection_name)
    
    db.add(db_dataset)

    from retriever.bm25_retriever import BM25Retriever
    bm25_corpus = []

    for split in splits:
        data_item = {}

        metadata = {}
        metadata['filename'] = file_name
        metadata['position'] = split.metadata
        
        data_item['content'] = split.page_content
        data_item['metadata'] =  metadata

        bm25_corpus.append(data_item)
    
    bm_retriever = BM25Retriever(txt_list=bm25_corpus)

    # 混合检索
    query = '该系统由什么科室负责发起的需求？'
    hybird_result = []
    
    db_results = db.search(query=query, top_k=2)
    for item in db_results:
        temp = {}
        temp['content'] = item['content']
        temp['metadata'] = item['metadata']
        hybird_result.append(temp)
    
    bm25_results = bm_retriever.search(query, top_n=2)
    for item in bm25_results:
        temp = {}
        temp['content'] = item['content']
        temp['metadata'] = item['metadata']
        hybird_result.append(temp)

    # reranker
    from reranker.reranker_bge_m3 import RerankerBGEM3
    reranker = RerankerBGEM3(model_id_key="BAAI/bge-m3-base-en-v1.5", device = 'cpu', is_api=False)
    reranker_results = reranker.rank(query, hybird_result)

    # llm 生成
    from src.llm.qwen_llm import QwenLLM
    URL_ADDRESS = ''
    llm = QwenLLM(url= URL_ADDRESS)
    
    PROMPT_TEMPLATE = """你被提供了1个问题，和根据这些问题检索到的文档，请分别依据检索内容和你自身的知识回答这些问题。

        问题：{question}

        检索到的文档：{search_documents}

        请给出你的回答（回答的文本写在<response></response>之间。
    """
    
    for item in reranker_results:
        context += item['content']
    prompt_text = PROMPT_TEMPLATE.format(
        question= query,
        search_documents= context
    )
    response = llm.generate(prompt_text)
    response = response.split('<response>')[-1].split('</response>')[0].strip()

    logger.info(response)
    return response
    # db_results = db_results['documents'][0]
    # db_results = ' '.join(db_results)
    

if __name__ == "__main__":
    main()