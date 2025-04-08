import os
import json
import random
from tqdm import tqdm

import embedding
from logger import logger
from src.retriever.chroma_retriever import ChromaRetriever
from src.splitter.recursive_char_spliter import RecursiveTextSplitter
from src.embedding.bge_embedding import BgeEmbedding
from src.embedding.remote_embedding import RemoteEmbedding

def read_all_files_in_directory(directory_path):
    data_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    data_list.append(line.strip())
    
    logger.info(f"read {len(data_list)} lines from {directory_path}")

    return data_list

def split_context(data_list, chunk_size=128, chunk_overlap=50):
    results = []
    r_spliter = RecursiveTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for instance in tqdm(data_list):
        splits = r_spliter.split_text(text=instance)
        results.extend(splits)

    logger.info(f"split {len(data_list)} lines to {len(results)} lines")

    return results

# def check_and_create_file(file_path):
#     if not os.path.exists(file_path):
#         with open(file_path, 'w') as file:
#             file.write('')  # 创建一个空文件
#         print(f"文件 {file_path} 已创建")
#     else:
#         print(f"文件 {file_path} 已存在")

def get_embedding(directory_path, embedding, save_dir):
    # 读取所有文件
    instances = read_all_files_in_directory(directory_path)

    # 分割上下文
    instances = split_context(instances)
    
    file_path = save_dir + '/crud_corpus.json'

    embeddings_list = []
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            embeddings_list = json.load(json_file)

    try:
        length = len(embeddings_list)

        # 从length开始继续获取embedding
        for instance in tqdm(instances[length: ]):
            metadata = {}
            metadata['filename'] = ''
            metadata['position'] = ''

            dic = {}
            dic['content'] = instance
            dic['embedding'] = embedding.get_embedding(instance)
            # dic['id'] = str(index)
            dic['metadata'] =  metadata
            embeddings_list.append(dic)
    except:
        logger.error(f"get embedding error")
        logger.error(f"finish {len(embeddings_list)} items")
        
        with open(file_path, "w") as json_file:
            json.dump(embeddings_list, json_file, ensure_ascii=False)
        
    return embeddings_list
    

def build_vector_db(directory_path, embedding, batch_size=256, persist_file_path=None, collection_name=None):
    db = ChromaRetriever(persist_file_path=persist_file_path, embedding=embedding)

    if db.load(collection_name=collection_name) == False:
        # 读取所有文件
        instances = read_all_files_in_directory(directory_path)

        # 分割上下文
        instances = split_context(instances)
        
        docs_file_path = directory_path + '/crud_corpus.json'

        # embeddings_list = []
        # for i in range(0, len(instances), batch_size):
        #     temp = instances[i: i + batch_size]
        #     embeddings_list.extend(embedding.get_batch_embedding(temp))
            
        docs = []
        # 如果docs文件不存在，则构建docs
        if not os.path.exists(docs_file_path):
            for index, instance in tqdm(enumerate(instances)):
                    metadata = {}
                    metadata['filename'] = ''
                    metadata['position'] = ''

                    dic = {}
                    dic['content'] = instance
                    dic['embedding'] = embedding.get_embedding(instance)
                    # dic['id'] = str(index)
                    dic['metadata'] =  metadata
                    docs.append(dic)

            # 保存docs到json文件    
            with open(docs_file_path, "w") as json_file:
                json.dump(docs, json_file, ensure_ascii=False)
        else:
            with open(docs_file_path, "r") as json_file:
                docs = json.load(json_file)

        db.create(collection_name=collection_name)
        db.add(docs)
        
    return db


def build_bm25_db():
    pass


def build_db(url):
    # embedding = BgeEmbedding(model_path = embedding_model, is_api=False)
    embedding = RemoteEmbedding(url)

    db = build_vector_db(directory_path=corpus_path, embedding=embedding, persist_file_path=persist_file_path, collection_name=collection_name)

