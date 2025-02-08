import os
import json
from typing import Dict, List, Optional, Tuple, Union
import requests

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .base_llm import BaseLLM

class QwenLLM(BaseLLM):
    def __init__(self, key: str = None, url: str = None, model_path: str = None, device: str = "cpu", max_new_tokens:int = 100, temperature:float = 1.0) -> None:
        super().__init__(key, url, model_path, device)

        if self.key != None:
            from openai import OpenAI 
            self.client = OpenAI(
                api_key= self.key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif self.url != None:
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
        elif self.model_path != None:
            # 从预训练模型加载因果语言模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,  # 模型标识符
                torch_dtype="auto",  # 自动选择张量类型
                device_map=self.device,  # 分布到特定设备上
                trust_remote_code=True  # 允许加载远程代码
            )
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,  # 分词器标识符
                trust_remote_code=True
            )
            
            # 加载配置文件
            self.config = AutoConfig.from_pretrained(
                self.model_path,  # 配置文件标识符
                trust_remote_code=True  # 允许加载远程代码
            )

            if self.device == "cpu":
                self.model.float()
            
            # 设置模型为评估模式
            self.model.eval()

    def generate(self, content: str) -> str:
        response = ''

        if self.key != None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
            completion = self.client.chat.completions.create(
                model="qwen-plus",
                messages=messages
            )
            response = completion.choices[0].message.content
        elif self.url != None:
            data = {'prompt': content, 'max_new_tokens': self.max_new_tokens, "temperature": self.temperature}

            response = requests.post(self.url, json=data)
            response = eval(response.text)['response']
            response = response[len(content):]
        elif self.model_path != None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response