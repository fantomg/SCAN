import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
df = pd.read_csv("E:edge_user_buy_package.csv")  # 替换为您的CSV文件路径

# 划分训练集和测试集+验证集
train_df, test_val_df = train_test_split(df, test_size=0.02, random_state=42)

# 再次划分测试集和验证集
test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

# 打印训练集、测试集和验证集的行数
print(f"训练集行数：{len(train_df)}")
print(f"测试集行数：{len(test_df)}")
print(f"验证集行数：{len(val_df)}")

# 保存为新的CSV文件
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
val_df.to_csv('val.csv', index=False)