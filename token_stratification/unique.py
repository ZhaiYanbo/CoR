import json

# 假设您的数据在这个文件中，每行是一个 JSON 对象
input_file = '/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples_1000.jsonl'
output_file = '/mnt/bn/volumezhaiyanbo/mlx/users/zhaiyanbo.188/hu/data/hard_samples_1000_unique.jsonl'

unique_tokens = set()

# 1. 读取并去重
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): # 跳过空行
                data = json.loads(line)
                if 'token' in data:
                    unique_tokens.add(data['token'])
    
    print(f"共提取到 {len(unique_tokens)} 个唯一 Token。")

    # 2. 保存为 JSON 列表
    # 注意：JSON 不支持直接存 set，需要先转为 list
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证特殊字符（如 Ġ）正常显示，不被转义
        json.dump(list(unique_tokens), f, ensure_ascii=False, indent=2) 
        
    print(f"保存成功：{output_file}")

except FileNotFoundError:
    print("未找到输入文件，请检查路径。")