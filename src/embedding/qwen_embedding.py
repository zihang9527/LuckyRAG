from typing import Dict, List, Optional, Tuple, Union
from .base_embedding import BaseEmbedding

class QwenEmbedding(BaseEmbedding):
    """
    A class for generating embeddings using the Qwen API.
    """
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", path: str = '', is_api: bool = True) -> None:
        """
        Initializes the OpenAIEmbedding object.

        :param api_key: API key for accessing the OpenAI API.
        :param base_url: Base URL for the OpenAI API.
        :param path: Path to any local resources (not used in this case).
        :param is_api: Flag indicating whether this is an API-based embedding.
        """
        super().__init__(path, is_api=is_api)  # Call the constructor of the base class

        from openai import OpenAI  # Importing the OpenAI client

        self.client = OpenAI(
            api_key = api_key,
            base_url= base_url
        )
        self.name = "qwen_api"  # Set the name of the embedding source

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using the OpenAI API.

        :param text: Text to embed.
        :return: A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        result = self.client.embeddings.create(model="text-embedding-v3", input= [text], encoding_format="float").data[0].embedding
        return result
