import json
from typing import Collection
from util import read_json_to_list
from embedding.qwen_embedding import QwenEmbedding
from llm.qwen_llm import QwenLLM
import math
from argument import Argument
import random
from retriever.chroma_retriever import ChromaRetriever
from logger import logger
from tqdm import tqdm
import os
import json

RAG_PROMPT_TEMPALTE="""参考信息：
{context}
---
我的问题或指令：
{question}
---
请根据上述参考信息回答和我的问题或指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来准确回答我的问题。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复。
你的回答:"""

def process_data_and_build_db(data_file_path, embedding, passage_num, noise_rate=0, persist_file_path=None, collection_name=None):
    instances = read_json_to_list(data_file_path)

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num
    
    parent_dir = os.path.dirname(data_file_path)
    docs_file_path = parent_dir + '/docs.json'
    docs = []

    # 如果docs文件不存在，则构建docs
    if not os.path.exists(docs_file_path):
        for instance in tqdm(instances):
            # 依据noiserate提取数量文本
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
            positive = instance['positive'][:pos_num]
            negative = instance['negative'][:neg_num]
            sents = positive + negative
            
            # 构建docs
            for sent in sents:
                dic = {}
                dic['content'] = sent
                dic['embedding'] = embedding.get_embedding(sent)
                docs.append(dic)

        with open(docs_file_path, "w") as json_file:
            json.dump(docs, json_file, ensure_ascii=False)
    else:
        with open(docs_file_path, "r") as json_file:
            docs = json.load(json_file)

    random.shuffle(docs)

    # 构建db
    db = ChromaRetriever(persist_file_path=persist_file_path)
    if db.load(collection_name=collection_name) == False:
        db.create(collection_name=collection_name, parse_output=docs)

    return db

def search(embedding, db, query, top_n=2):
    query_emb = embedding.get_embedding(query)
    result = db.search(query_emb, top_n = top_n)

    temp = []
    for ls in result['documents']:
        for s in ls:
            temp.append(s)
    context = "\n".join(temp)

    return context

# 检查一个query的答案是否在ground_truth中
def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()

    if type(ground_truth) is not list:
        ground_truth = [ground_truth]

    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance)  == list:
            flag = False 
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels


def eval():
    args = Argument().args

    embedding = QwenEmbedding(api_key=args.api_key, is_api=True)
    llm = QwenLLM(model_id_key=args.api_key, is_api=True)

    db = process_data_and_build_db(args.input_file, embedding, args.passage_num, args.noise_rate, args.persist_file_path, collection_name=args.collection_name)
    
    instances = read_json_to_list(args.input_file)
    
    labels = []
    results = []
    parent_dir = os.path.dirname(args.input_file)
    result_file_path = parent_dir + '/result.json'
    
    # 如果result文件存在，则直接读取
    if not os.path.exists(result_file_path):
        for instance in tqdm(instances):
            query = instance['query']
            answer = instance['answer']

            context = search(embedding, db, query, args.top_n)
            
            prompt_text = RAG_PROMPT_TEMPALTE.format(
                context=context,
                question=query,
            )

            try:
                llm_result = llm.generate(prompt_text)
            # print(llm_result)
                logger.info(f"llm_result: {llm_result}")
            except Exception as e:
                print(e)

                llm_result = ''
                logger.info(f"prompt_text: {prompt_text}")

            label = checkanswer(llm_result, answer)
            labels.append(label)

            temp = {}
            temp['query'] = query
            temp['answer'] = answer
            temp['llm_result'] = llm_result 
            temp['label'] = label
            results.append(temp)
        
        with open(result_file_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False)
    else:
        with open(result_file_path, 'r') as f:
            results = json.load(f)
            for result in results:
                labels.append(result['label'])

    total = len(labels)
    count = 0
    for label in labels:
        if 1 in label and 0 not in label:
            count += 1
    acc = count / total
    logger.info(f"acc: {acc}")

if __name__ == '__main__':
    eval()
    