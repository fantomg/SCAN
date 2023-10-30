import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
data = pd.read_csv("E:edge_user_buy_package.csv")

# 将数据帧划分为训练集和数据集
train_data, test_data = train_test_split(data, test_size=0.004)

# 将训练集保存为新的CSV文件
train_data.to_csv('train_data.csv', index=False)

# 将测试集保存为新的CSV文件
test_data.to_csv('test_data.csv', index=False)
