from logger import logger

def main():
    ocr_model_path = './data/ocr_model'
    artifacts_path = './data/artifacts'
    pdf_file_path = './data/test.pdf'
    embedding_model = ''
    persist_file_path = ''
    collection_name = 'test'
    bm25_base_dir="data/db/bm_corpus"
    bm25_db_name = "bm25_data"
    query = '该系统由什么科室负责发起的需求？'
    rerank_model_path = ''
    URL_ADDRESS = ''
    
    pdf_parser = PdfParser(ocr_model_path=ocr_model_path, artifacts_path=artifacts_path)
    parse_result = pdf_parser.parse(file_path_or_url=pdf_file_path)
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
    embedding = BgeEmbedding(model_path = embedding_model, is_api=False)

    db_dataset = []
    for split in splits:
        data_item = {}

        metadata = {}
        metadata['filename'] = file_name
        metadata['position'] = list(split.metadata.keys())[0]
        
        data_item['content'] = split.page_content
        data_item['metadata'] =  metadata
        data_item['embedding'] = embedding.get_embedding(data_item['content'])

        db_dataset.append(data_item)
    
    print(len(db_dataset))

    from src.retriever.chroma_retriever import ChromaRetriever
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
        metadata['position'] = list(split.metadata.keys())[0]
        
        data_item['content'] = split.page_content
        data_item['metadata'] =  metadata

        bm25_corpus.append(data_item)

    bm_retriever = BM25Retriever(base_dir=bm25_base_dir, db_name = bm25_db_name)
    if bm_retriever.load_bm25_data() == False:
        bm_retriever.build(bm25_corpus)
    else:
        bm_retriever.add(bm25_corpus)
    
    # 混合检索
    hybird_result = []

    db_results = db.search(query=query, top_n=2)
    
    for document, metadata in zip(db_results['documents'][0], db_results['metadatas'][0]):
        temp = {}
        temp['content'] = document
        temp['metadata'] = metadata
        hybird_result.append(temp)
    
    bm25_results = bm_retriever.search(query, top_n=2)
    for item in bm25_results:
        temp = {}
        temp['content'] = item[1]['content']
        temp['metadata'] = item[1]['metadata']
        hybird_result.append(temp)

    # reranker
    from src.reranker.reranker_bge_m3 import RerankerBGEM3
    reranker = RerankerBGEM3(model_id_key= rerank_model_path, device = 'cpu', is_api=False)
    reranker_results = reranker.rank(query, hybird_result)

    # llm 生成
    from src.llm.qwen_llm import QwenLLM
    llm = QwenLLM(url= URL_ADDRESS)
    
    PROMPT_TEMPLATE = """你被提供了1个问题，和根据这些问题检索到的文档，请分别依据检索内容和你自身的知识回答这些问题。

        问题：{question}

        检索到的文档：{search_documents}

        请给出你的回答（回答的文本写在<response></response>之间。
    """
    
    context = ''
    for item in reranker_results:
        context += item[1]['content']
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