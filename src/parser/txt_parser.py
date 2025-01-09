from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import sys
import os
import re

from nltk.tokenize import sent_tokenize
from .base_parser import BaseParser

class TXTParser(BaseParser):
    """
    Parser for txt files
    """
    type = 'txt'
    def __init__(self, file_path: str=None, model=None) -> None:
        super().__init__(file_path, model)
        
    def parse(self) -> List[Dict]:
        page_sents = self._to_sentences()

        if not page_sents:
            return None
        
        self.parse_output = []
        for _, sent in page_sents:
            file_dict = {}
            file_dict['title'] = None
            file_dict['author'] = None
            file_dict['page'] = None
            file_dict['content'] = sent
            file_dict['embedding'] = self.get_embedding(sent)
            file_dict['file_path'] = self.file_path
            file_dict['subject'] = None
            
            self.parse_output.append(file_dict)
        
        return self.parse_output

    def _to_sentences(self) -> List[Tuple[int, str]]:
        """
        Parse txt file to text [(pageno, sentence)]
        """
        if not self._check_format():
            self.parse_output = None
            return []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # remove hyphens 
        raw_text = re.sub(r"-\n(\w+)", r"\1", raw_text)
        raw_text = raw_text.replace("\n", " ")
        return list(map(lambda x: (0, x), sent_tokenize(raw_text)))

    
    @property
    def metadata(self) -> defaultdict:
        # txt files don't have metadata
        if not self._metadata:  
            metadata = defaultdict(str)  
            self._metadata = metadata  
  
        return self._metadata  


    def _check_format(self) -> bool:
        f_path: Path = Path(self.file_path)
        return f_path.exists() and f_path.suffix == '.txt'


if __name__ == "__main__":
    parser = TXTParser(sys.argv[1], None)
    print(parser._to_sentences())
    # print(parser.parse_output)
    # parser.parse()
    
    # import os
    # current_path = os.getcwd()
    # print("当前路径：", current_path)
    
    # parser = TXTParser('LuckyRAG/data/data.txt', None)
    # print(parser._to_sentences())
    # parser.parse()
    # print(parser.parse_output)