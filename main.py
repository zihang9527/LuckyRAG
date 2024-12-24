import os
import json

from embedding.qwen_embedding import QwenEmbedding
from parser.txt_parser import TXTParser

current_path = '/cloudide/workspace/LuckyRAG'
os.chdir(current_path)
file_path = 'data/data.txt'

embedding = QwenEmbedding(api_key='sk-effb7a023271466c9ceb33365c970e5b')

# parser = TXTParser(file_path=file_path, model=embedding)
# parser.parse()
# print(parser.parse_output[0])
# print(len(parser.parse_output))

# 解析文件
# parser.parse_file('data/余大伟.txt')
# 初始化向量存储
# embedding = QwenEmbedding()
# 构建向量数据库
# embedding.build_db(parser.get_texts())
# 构建查询

