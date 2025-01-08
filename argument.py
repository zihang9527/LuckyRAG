import argparse

class Argument: 
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--input_file', type = str, default='', help='输入文件的路径')
        self.parser.add_argument('--embedding_model', type = str, default='', help='embedding model')
        self.parser.add_argument('--llm_model', type = str, default='', help='llm_model')
        self.parser.add_argument('--persist_file_path', type=str, default='', help='db saved path')
        self.parser.add_argument('--collection_name', type=str, default='', help='')
        self.parser.add_argument('--api_key', type=str, default='', help='api_key')
        self.parser.add_argument('--passage_num', type=int, default=5, help='')
        self.parser.add_argument('--noise_rate', type=float, default=0.2, help='')
        self.parser.add_argument('--top_n', type=int, default=3, help='the top of query result from the db')

        self.args = self.parser.parse_args()