import chromadb
from typing import Dict, List, Optional, Tuple, Union

class ChromaRetriever:
    def __init__(self, persist_file_path: str = None, embedding = None) -> None:
        if persist_file_path != None:
            self.client = chromadb.PersistentClient(path=persist_file_path)
        else:
            self.client = chromadb.Client()
        self.collection = ''
        self.embedding = embedding

    def create(self, collection_name: str):
        self.collection = self.client.create_collection(name=collection_name)

    def add(self, parse_output: List[Dict]):
        for id, dic in enumerate(parse_output):
            self.collection.add(
                documents=[dic['content']],
                embeddings=[dic['embedding']],
                ids= str(id))

    def load(self, collection_name: str):
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print('collection load success.')
            return True
        except:
            print('collection not exist. Please create first.')
            return False
    
    def search(self, query: str, top_n: int = 3):
        query_embedding = self.embedding.get_embedding(query)

        db_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )
        return db_results
        
        