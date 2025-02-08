# # FILEPATH: \cloudide\workspace\LuckyRAG\qwen_infer.py
# import time
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # 加载模型和分词器
# model_id = ""
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval()

# # 输入文本
# input_text = "你好，我是人工智能助手，有什么可以帮到您？"

# # 编码输入文本
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# # 确保模型在正确的设备上
# input_ids = input_ids.to(model.device)

# # 进行推理并测量时间
# start_time = time.time()
# with torch.no_grad():
#     outputs = model.generate(input_ids, max_new_tokens=512)
# end_time = time.time()

# # 计算推理时间
# inference_time = end_time - start_time

# # 解码生成的文本
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 打印结果
# print(f"Generated text: {generated_text}")
# print(f"Inference time: {inference_time} seconds")