from re import NOFLAG
from langchain_text_splitters import MarkdownHeaderTextSplitter

class MDSpliter:
    def __init__(self, headers_to_split_on, strip_headers = False):
        self.headers_to_split_on = headers_to_split_on
        self.strip_headers = strip_headers

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=strip_headers
        )
            
    def split(self, markdown_document) -> list:
        md_header_splits = self.markdown_splitter.split_text(markdown_document)

        return md_header_splits
