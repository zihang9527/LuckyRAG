import os
import json

def read_json_to_list(file_path):
    """
    从JSON文件读取数据到列表中。

    参数:
    file_path (str): JSON文件的路径

    返回:
    list: 包含JSON文件中数据的列表
    """
    result = []
    with open(file_path,'r') as f:
        for line in f:
            result.append(json.loads(line))
    
    return result


def write_list_to_json(data, file_path):
    """
    将列表数据写入JSON文件。

    参数:
    data (list): 需要写入JSON文件的列表数据
    file_path (str): 目标JSON文件的路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    result = read_json_to_list('data/zh_refine.json')
    print(result[0])