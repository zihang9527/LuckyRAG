from multiprocessing import context
from typing import Collection
import math
import random
from tqdm import tqdm
import os
import json
import datetime
import time

from logger import logger
from argument import Argument
from src.embedding.bge_embedding import BgeEmbedding
from src.llm.qwen_llm import QwenLLM
from src.retriever.chroma_retriever import ChromaRetriever
from metric import (
    bleu_score, 
    rougeL_score, 
)

PROMPT_TEMPLATE = """你是一位新闻编辑，现在，你被提供了1个问题，和根据这些问题检索到的文档，请分别检索内容和你自身的知识回答这些问题。以下是个例子：

问题：上海和成都市体育局在促进体育消费和全民健身运动方面有哪些相似和不同的措施？

检索文档: 在第15个全民健身日来临之际，上海市体育局将联合美团、大众点评发放500万元体育消费券，3000多家上海本地运动门店参与其中，共同点燃全民健身运动热情，促进体育消费增长。▲8月5日上午10点，上海市体育局将联合美团、大众点评发放新一轮体育消费券2023年上海体育消费券以“全民优惠健身，共享美好生活”为主题，在8月5日-9月3日期间分四期进行发放。第一期消费券发放时间为8月5日10：00-8月13日24：00，第二期消费券发放时间为8月14日-8月20日，第三期8月21日-8月27日，第四期8月28日-9月3日。实时定位在上海的消费者，可以在发放时间内进入美团、大众点评App，搜索“上海体育消费券”进行领取。为满足消费者更多个性化的需求，本轮体育消费券活动准备了满200减80、满120减50、满60减30、满40减20、满20减10和满10减5共六个面额的消费券，消费者可按需领用，先到先得。每位消费者每期最多可领取3张消费券，且每位消费者同时最多可持有3张。据“上海体育”公众号介绍，本次体育消费券适用场景多、覆盖范围广、优惠力度大。在发布会上，成都市体育局副局长陈志介绍，以成都大运会筹办举办为契机，成都积极开展“爱成都·迎大运”“运动成都·悦动生活”“万千商家齐参与”等主题体育消费促进活动，发放各类体育消费券和惠民运动券，促进体育消费持续稳步增长。2022年成都体育消费总规模为578.6亿元，居民人均体育消费为2720.6元。      ▲8月4日，成都大运会体操项目女子个人全能决赛看台上，观众为比赛队员加油 资料配图 摄影 陶轲  为持续激发体育消费活力和增长潜力，下一步，成都将持续深化体育消费试点工作，积极推进体育消费提质扩容。启动户外运动季活动，发布十大最受欢迎时尚运动消费场景。  具体而言，陈志介绍说，成都将加快推动“体育＋会展＋消费”平台建设，办好中国（成都）生活体育大会、“巴山蜀水·运动川渝”体育旅游休闲消费季、世界赛事名城发展大会、中国国际体育用品博览会等重大体育展会活动，为城市体育消费增长提供更多资源链接。

回答：上海市体育局联合美团、大众点评发放了总额500万元的体育消费券，覆盖3000多家本地运动门店，并设置了不同面额的消费券供消费者领取。而成都市体育局则是利用成都大运会的契机发放各类体育消费券和惠民运动券，同时计划通过举办大型体育展会活动和推动“体育＋会展＋消费”平台建设来进一步促进体育消费的提质扩容。

问题：{question}

检索到的文档：{search_documents}

请给出你的回答（回答的文本写在<response></response>之间。
"""

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

def process_data_and_build_db(directory_path, embedding, persist_file_path=None, collection_name=None):
    db = ChromaRetriever(persist_file_path=persist_file_path, embedding=embedding)

    if db.load(collection_name=collection_name) == False:
        # 读取所有文件
        instances = read_all_files_in_directory(directory_path)
        docs_file_path = directory_path + '/docs.json'
        docs = []
        # 如果docs文件不存在，则构建docs
        if not os.path.exists(docs_file_path):
            for instance in tqdm(instances):
                    dic = {}
                    dic['content'] = instance
                    dic['embedding'] = embedding.get_embedding(instance)
                    docs.append(dic)

            with open(docs_file_path, "w") as json_file:
                json.dump(docs, json_file, ensure_ascii=False)
        else:
            with open(docs_file_path, "r") as json_file:
                docs = json.load(json_file)

        db.create(collection_name=collection_name)
        db.add(docs)
        
    return db

def get_task_datasets(file_path: str)-> list:
    task_datasets = []
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_datasets += data['questanswer_1doc']
    task_datasets += data['questanswer_2docs']
    task_datasets += data['questanswer_3docs']
    
    random.shuffle(task_datasets)

    return task_datasets

def retrieve_docs(query, retriever, top_n):
    query_text = query["questions"]
    db_results = retriever.search(query=query_text, top_n=top_n)
    db_results = db_results['documents'][0]
    db_results = ' '.join(db_results)

    return db_results

def model_generation(obj, llm):
    prompt_text = PROMPT_TEMPLATE.format(
        question= obj["questions"],
        search_documents= obj["retrieve_context"],
    )
    res = llm.generate(prompt_text)
    real_res = res.split('<response>')[-1].split('</response>')[0]

    return real_res.strip()

def scoring(data_point: dict) -> dict:
    generated_text = data_point["generated_text"]
    ground_truth_text = data_point["answers"]
    data_point["ground_truth_text"] = ground_truth_text
    
    QA_avg_F1, QA_recall, quest_eval_save = 0.0, 0.0, {}
    bertscore = 0.0
    
    bleu_avg, bleu1, bleu2, bleu3, bleu4 = bleu_score(generated_text, ground_truth_text)

    return {
        'metrics': {
            'bleu-avg': bleu_avg or 0.0,
            'bleu-1': bleu1 or 0.0,
            'bleu-2': bleu2 or 0.0,
            'bleu-3': bleu3 or 0.0,
            'bleu-4': bleu4 or 0.0,
            'rouge-L': rougeL_score(generated_text, ground_truth_text) or 0.0,
            'bertScore': bertscore,
            'QA_avg_F1': QA_avg_F1,
            'QA_recall': QA_recall,
            'length': len(generated_text)
        },
        'log': {
            'generated_text': generated_text,
            'ground_truth_text': ground_truth_text,
            'quest_eval_save': quest_eval_save,
            'evaluateDatetime': str(datetime.datetime.now()),
        },
        'valid': len(generated_text.strip()) != 0
    }

def compute_overall(results: list[dict]) -> dict:
    overall = {'bleu-avg': 0, 'bleu-1': 0, 'bleu-2': 0, 'bleu-3': 0, 
                'bleu-4': 0, 'rouge-L': 0, 'bertScore': 0, 'QA_avg_F1': 0, 
                'QA_recall': 0, 'length': 0}
    
    for result in results:
        overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}

    overall_save = {f'avg. {key}': value / len(results) for key, value in overall.items() if key != 'QA_avg_F1' and key != 'QA_recall'}
    overall_save['num'] = len(results)
    
    return overall_save

def eval():
    # args = Argument().args
    embedding_model = ''
    llm_path = ''
    doc_path = ''
    persist_file_path = ''
    collection_name = ''
    dataset_file = '/home/ubuntu/llm/eval/eval_data/eval_data.json'
    result_file_path = 'res/eval_result.json'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    embedding = BgeEmbedding(model_path = embedding_model, is_api=False)
    llm = QwenLLM(model_id_key=llm_path, device=device, is_api=False)

    db = process_data_and_build_db(doc_path, embedding, persist_file_path, collection_name=collection_name)

    task_datasets = get_task_datasets(dataset_file)
    logger.info(f"task_datasets len: {len(task_datasets)}")
    
    results = []
    for data in tqdm(task_datasets):

        start_time = time.time()
        db_results = retrieve_docs(data, db, top_n=5)
        end_time = time.time()
        print(f"retrieve time: {end_time - start_time}")

        data["retrieve_context"] = db_results

        start_time = time.time()
        generated_text = model_generation(data, llm)
        end_time = time.time()
        print(f"generate time: {end_time - start_time}")
        
        data["generated_text"] = generated_text

        result = scoring(data)
        results.append(result)

        # logger.info(f"result: {result}")

    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    
    overall = compute_overall(results)
    logger.info(f"overall: {overall}")

if __name__ == '__main__':
    # eval()
    print("hello world")