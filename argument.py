import argparse

class Argument: 
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--input_file', type = str, help='输入文件的路径')
        self.parser.add_argument('--persist_file_path', type=str, help='db saved path')
        self.parser.add_argument('--collection_name', type=str, help='')
        self.parser.add_argument('--api_key', type=str, help='api_key')
        self.parser.add_argument('--passage_num', type=int, help='')
        self.parser.add_argument('--noise_rate', type=float, help='')
        self.parser.add_argument('--top_n', type=int, help='the top of query result from the db')

        self.args = self.parser.parse_args()