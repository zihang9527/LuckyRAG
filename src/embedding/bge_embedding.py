from typing import Dict, List, Optional, Tuple, Union
from .base_embedding import BaseEmbedding
from transformers import AutoTokenizer, AutoModel
import torch

class BgeEmbedding(BaseEmbedding):
    """
    A class for generating embeddings using the Bge embedding.
    """
    def __init__(self, model_path, is_api: bool = False) -> None:
        super().__init__(model_path, is_api=is_api)  # Call the constructor of the base class

        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.eval()

    def get_embedding(self, text: str) -> List[float]:
        '''
        Generates an embedding for the given text using the Bge embedding.
        '''
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]

        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = sentence_embeddings.tolist()[0]

        # print("Sentence embeddings:", sentence_embeddings)

        return sentence_embeddings
