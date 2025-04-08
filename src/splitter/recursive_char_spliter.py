from langchain_text_splitters import RecursiveCharacterTextSplitter

class RecursiveTextSplitter:
    def __init__(self, chunk_size = 250, chunk_overlap = 15):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        
    def split_documents(self, document) -> list:
        splits = self.text_splitter.split_documents(document)

        return splits

    def split_text(self, text) -> list:
        splits = self.text_splitter.split_text(text)

        return splits