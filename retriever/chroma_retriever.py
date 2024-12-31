import chromadb
from typing import Dict, List, Optional, Tuple, Union

class ChromaRetriever:
    def __init__(self, persist_file_path: str = None) -> None:
        if persist_file_path != None:
            self.client = chromadb.PersistentClient(path=persist_file_path)
        else:
            self.client = chromadb.Client()
        self.collection = ''

    def create(self, collection_name: str, parse_output: List[Dict]):
        self.collection = self.client.create_collection(name=collection_name)

        for id, dic in enumerate(parse_output):
            self.collection.add(
                documents=[dic['content']],
                embeddings=[dic['embedding']],
                ids= str(id)
            )

    def load(self, collection_name: str):
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print('collection load success.')
            return True
        except:
            print('collection not exist. Please create first.')
            return False
    
    def search(self, query_embedding: list, top_n: int = 3):
        db_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n
        )
        return db_results
        
        