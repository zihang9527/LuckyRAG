import os
import json
from typing import Dict, List, Optional, Tuple, Union

from embedding.qwen_embedding import QwenEmbedding
from llm.qwen_llm import QwenLLM
from parser.txt_parser import TXTParser

import chromadb

api_key = ''
embedding = QwenEmbedding(api_key=api_key)

file_path = 'data/data.txt'
parser = TXTParser(file_path=file_path, model=embedding)
parser.parse()

# print(parser.parse_output[0])
# print(len(parser.parse_output))

client = chromadb.Client()
test_collection = client.create_collection(name="test")

for id, dic in enumerate(parser.parse_output):
    test_collection.add(
        documents=[dic['content']],
        embeddings=[dic['embedding']],
        ids= str(id)
    )

query = 'what is the Qwen2.5-math model?'
qwen = QwenLLM(model_id_key=api_key, is_api=True)
qwen_result = qwen.generate(query)
print(qwen_result)

embeddings_query = embedding.get_embedding(query)

db_results = test_collection.query(
    query_embeddings=[embeddings_query],
    n_results=2
)
temp = []
for ls in db_results['documents']:
    for s in ls:
        temp.append(s)
context = "\n".join(temp)

RAG_PROMPT_TEMPALTE="""参考信息：
{context}
---
我的问题或指令：
{question}
---
我的回答：
{answer}
---
请根据上述参考信息回答和我的问题或指令，修正我的回答。前面的参考信息和我的回答可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你修正的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你修正的回答:"""

prompt_text = RAG_PROMPT_TEMPALTE.format(
        context=context,
        question=query,
        answer=qwen_result
    )
qwen_result = qwen.generate(prompt_text)
print(qwen_result)


# 解析文件
# parser.parse_file('data/余大伟.txt')
# 初始化向量存储
# embedding = QwenEmbedding()
# 构建向量数据库
# embedding.build_db(parser.get_texts())
# 构建查询

