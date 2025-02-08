import os
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Base class for embeddings
    """
    def __init__(self, key: str = None, url: str = None, model_path: str = None, device: str = "cpu") -> None:
        super().__init__()
        self.model_path = model_path
        self.key = key
        self.url = url
        self.device = device

    @abstractmethod
    def generate(self, content: str) -> str:
        raise NotImplemented