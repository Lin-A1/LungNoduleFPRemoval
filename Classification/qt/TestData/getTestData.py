import os
import pandas as pd
import shutil

# 读取CSV文件
candidates_df = pd.read_csv(r'/data/CSVFILES/candidates.csv')

# 筛选class列中值为1的数据
class_1_candidates = candidates_df[candidates_df['class'] == 1]

# 获取标识名为seriesuid列的数据
seriesuid_column = class_1_candidates['seriesuid']

# 匹配文件夹中相应的文件名
data_folder = r'E:\code\pycharm_code\DeepLearning\LungNoduleFPRemoval\data'
matching_files = []
for seriesuid in seriesuid_column:
    file_path = os.path.join(data_folder, seriesuid + '.mhd')  # 假设文件扩展名为.mhd
    if os.path.exists(file_path):
        matching_files.append(file_path)


# 输出匹配的文件列表
print("匹配的文件列表:")
for file in matching_files:
    print(file)

# 获取文件名
file_name = matching_files[-7].split('\\')[-1][:-4]

# 从数据文件夹复制文件到当前目录
for ext in ['.mhd', '.raw']:
    src_file = os.path.join(data_folder, file_name + ext)
    dst_file = os.path.join('/', file_name + ext)
    shutil.copyfile(src_file, dst_file)
    print(f"复制文件 {src_file} 到 {dst_file}")

# 获取匹配文件对应的行
matching_row = candidates_df[candidates_df['seriesuid'] == os.path.basename(matching_files[-7])[:-4]]

print("匹配的行:")
print(matching_row)

class_1_df = matching_row[matching_row['class'] == 1]

# 保留 class 为 0 的 20 个随机行
class_0_df = matching_row[matching_row['class'] == 0].sample(n=20, random_state=42)

# 合并两个 DataFrame
matching_row = pd.concat([class_1_df, class_0_df])

# 将匹配到的行写入CSV文件
matching_row.to_csv('./test.csv', index=False)
