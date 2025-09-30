'''
将原始的jsonl文件转换格式并拆分成训练集和测试集
'''
import os.path
import random
import json
import os
# 设置随机种子以确保可重复性
random.seed(42)
PROMPT = "你是一个法律专家，你需要根据用户的问题，给出准确的回答。"
MAX_LENGTH = 2048

def load_shuffle_transfer_data(origin_path):
    with open(origin_path, 'r', encoding='utf-8') as f:
        try:
            data_list = json.load(f)
        except Exception as e:
            print(f"加载数据失败:{e}")
        print(f"原始数据集数量: {len(data_list)}")
        random.shuffle(data_list)
        print("数据已随机打乱")
        messages = []
        for line in data_list:
            input = line["input"]
            output = line["output"]
            message = {
                "instruction": PROMPT,
                "input": input,
                "output": output
            }
            messages.append(message)
        split_ratio = 0.9
        split_index = int(len(messages) * split_ratio)
        train_data = messages[:split_index]
        val_data = messages[split_index:]
        return train_data, val_data


def process_save_data(origin_path):
    if not os.path.exists(origin_path):
        print("目录不存在无法读取文件")
    save_dir = os.path.dirname(origin_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"创建保存目录{save_dir}成功")
    train_path = os.path.join(save_dir, "train_data.jsonl")
    val_path = os.path.join(save_dir, "val_data.jsonl")
    train_data, val_data = load_shuffle_transfer_data(origin_path)
    with open(train_path, 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    with open(val_path, 'w', encoding='utf-8') as f:
        for line in val_data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"保存文件到{train_path},{val_path}成功,数据量分别为{len(train_data)},{len(val_data)}")



process_save_data('./CrimeKgAssitant.json')

啥都不要生成我存个信息