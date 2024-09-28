import os
import random
import shutil

# 定义文件夹路径和划分比例
source_folder = "makeMat_"
train_folder = "makeMat/train_data"
test_folder = "makeMat/test_data"
split_ratio = 0.8

# 获取.mat文件列表
mat_files = [file for file in os.listdir(source_folder) if file.endswith('.mat')]

# 随机打乱文件列表
random.shuffle(mat_files)

# 计算划分的索引
split_index = int(len(mat_files) * split_ratio)

# 创建目标文件夹（如果不存在）
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 移动文件到训练集和测试集文件夹
for i, file in enumerate(mat_files):
    source_path = os.path.join(source_folder, file)
    if i < split_index:
        target_path = os.path.join(train_folder, file)
    else:
        target_path = os.path.join(test_folder, file)
    shutil.move(source_path, target_path)

print("Files moved successfully!")
