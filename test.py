import sys
from logger import logger

if __name__ == '__main__':
    import os.path

    # 使用示例
    file_path = "path/to/your/file.txt"  # 替换为你要获取上级目录的文件路径
    parent_dir = os.path.dirname(file_path)
    print(f"文件 {file_path} 的上级目录是: {parent_dir}")
    print(parent_dir+'/1.txt')
    