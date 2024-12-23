import os
from openai import OpenAI, embeddings
import json
import chromadb

# chroma
client = chromadb.Client()
collection = client.create_collection(name="my_collection")

# qwen api
llm = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key = "sk-effb7a023271466c9ceb33365c970e5b",
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# build db
texts = ["余大伟出生于1997年，今年2024年，它今年28岁。",'余大伟最喜欢打游戏。','余大伟喜欢白色双马尾、穿着白色丝袜的萝莉，男女不限']

embeddings_response = llm.embeddings.create(
    model="text-embedding-v3",
    input= texts,
    encoding_format="float"
)
for i in range(len(embeddings_response.data)):
    embeddings = embeddings_response.data[i].embedding
    collection.add(
        documents=[texts[i]],
        embeddings=[embeddings],
        ids=[f"id{i+1}"]
    )

query = ['余大伟喜欢什么']
embeddings_query = llm.embeddings.create(
    model="text-embedding-v3",
    input= query,
    encoding_format="float"
)
query_embedding = embeddings_query.data[0].embedding
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

res = ''
for i in range(len(results['ids'][0])):
    res += results['documents'][0][i]
    res += '\n'
print(res)
