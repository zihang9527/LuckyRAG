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

import sys
import random
import time
import json
from pathlib import Path
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

import embedding
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import logger

# chunk md file
def split(directory_path, chunk_size = 250, chunk_overlap = 15):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                page_content = f.read()

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(page_content)
        # print(md_header_splits)

        chunk_size = 250
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(md_header_splits)


def main(ouput_type):
    # 
    ocr_model_path = ''
    easy_ocr_option = EasyOcrOptions()
    easy_ocr_option.model_storage_directory = ocr_model_path
    easy_ocr_option.lang = ['en', 'ch_sim']
    
    artifacts_path = ""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.artifacts_path = artifacts_path
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE  # use more accurate TableFormer model
    pipeline_options.ocr_options  = easy_ocr_option

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
    
    start_time = time.time()
    result = doc_converter.convert(source)
    end_time = time.time() - start_time
    logger.info(f"Document converted in {end_time:.2f} seconds.")
    # print(result.document.export_to_markdown())

    ## Export results
    output_dir = Path("./data/docling")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = result.input.file.stem

    if ouput_type == 'json':
        # Export Deep Search document JSON format:
        with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
            fp.write(json.dumps(result.document.export_to_dict()))
    elif ouput_type == 'text':
        # Export Text format:
        with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
            fp.write(result.document.export_to_text())
    elif ouput_type == 'md':
        # Export Markdown format:
        with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
            fp.write(result.document.export_to_markdown())
    elif ouput_type == 'doctags':
    # Export Document Tags format:
        with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
            fp.write(result.document.export_to_document_tokens())



if __name__ == '__main__':
    # main()
    from src.parser.pdf_parser import PdfParser

    pdf_parser = PdfParser(ocr_model_path='./data/ocr_model', artifacts_path='./data/artifacts')
    parse_result = pdf_parser.parse(file_path_or_url='./data/test.pdf')
    
    md_result = pdf_parser.convert_to_file(parse_result, ouput_type='md')

    # from langchain_community.document_loaders import UnstructuredMarkdownLoader
    # markdown_path = "../../../../../README.md"
    # loader = UnstructuredMarkdownLoader(markdown_path)
    # data = loader.load()

    from src.splitter.md_spliter import MDSpliter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_spliter = MDSpliter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    splits = md_spliter.split(markdown_document = parse_result)

    from src.splitter.recursive_char_spliter import RecursiveTextSplitter
    chunk_size = 250
    chunk_overlap = 15
    r_spliter = RecursiveTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    splits = r_spliter.split(document = splits)

    for i, split in enumerate(splits):
        
        metadata = {}
        metadata['filename'] = parse_result.document.origin.filename
        metadata['position'] = splits[0].metadata
    
        dic = {}
        embedding = ''
        dic['content'] = splits[0].page_content
        dic['metadata'] =  metadata
        dic['embedding'] = embedding
    
    # split(directory_path='./data/docling')
    
    # from src.splitter.recursive_char_spliter import RecursiveTextSplitter

    # # 初始化RecursiveTextSplitter
    # splitter = RecursiveTextSplitter(chunk_size=250, chunk_overlap=15)

    # # 读取Markdown文件的内容
    # with open('./data/docling/test.md', 'r', encoding='utf-8') as file:
    #     markdown_content = file.read()

    # # 分割Markdown内容
    # splits = splitter.split(markdown_content)

    # # 打印分割后的文本块
    # for i, split in enumerate(splits):
    #     print(f"Split {i + 1}:\n{split}\n")

    # # 打印分割后的文本块数量
    # print(f"Number of splits: {len(splits)}")
    
    # import os.path
    # 使用示例
    # file_path = "path/to/your/file.txt"  # 替换为你要获取上级目录的文件路径
    # parent_dir = os.path.dirname(file_path)
    # print(f"文件 {file_path} 的上级目录是: {parent_dir}")
    # print(parent_dir+'/1.txt')

    # from retriever.bm25_retriever import BM25Retriever
    # corpus = [
    #     "Hello there good man!",
    #     "It is quite windy in London",
    #     "How is the weather today?"
    # ]
    # bm_retriever = BM25Retriever(txt_list=corpus)
    # query = "windy London"
    # results = bm_retriever.search(query, top_n=3)

    # from reranker.reranker_bge_m3 import RerankerBGEM3

    # reranker = RerankerBGEM3(model_id_key="BAAI/bge-m3-base-en-v1.5", device = 'cpu', is_api=False)

    # reranker_results = reranker.rank(query, [result[1] for result in results])
    # print(reranker_results)
    # pass

    # import requests
    # address = ''
    # url = ''
    # # Flask服务器的地址和端口
    # flask_server_url = 'http://localhost:5000'  # 请替换为你的Flask服务器实际地址和端口

    # # 发送GET请求
    # data = {'prompt': '你是谁'}
    # response = requests.post(flask_server_url, data=data)

    # # 打印响应内容
    # print(response.text)

    # import easyocr
    # ocr_model_path = ''
    # reader = easyocr.Reader(lang_list = ['en', 'ch_sim'], model_storage_directory=ocr_model_path)
    # result = reader.readtext("test.png")

    # output: ## Docling Technical Report [...]"

    