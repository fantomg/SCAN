import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib

# 读取训练集数据
train_data = pd.read_csv('test.csv', header=None)

# 随机取样一万条数据
train_data = train_data.sample(n=30000, random_state=42)

# 分割特征和标签
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

# 创建朴素贝叶斯分类器模型
model = GaussianNB()

# 拟合模型
model.fit(X, y)

# 保存模型
joblib.dump(model, 'model.pkl')