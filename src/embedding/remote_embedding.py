from typing import Dict, List, Optional, Tuple, Union
from .base_embedding import BaseEmbedding
import requests

class RemoteEmbedding(BaseEmbedding):
    """
    A class for generating embeddings using the Bge embedding.
    """
    def __init__(self, url, is_api: bool = False) -> None:
        super().__init__(url, is_api=is_api)  # Call the constructor of the base class
        self.url = url

    def get_embedding(self, text: str) -> List[float]:
        data = {'content': text}
        response = requests.post(self.url, json=data)
        response = eval(response.text)['response']
        
        return response

    def get_batch_embedding(self, batch: List[str]) -> List[List[float]]:
        pass
    
