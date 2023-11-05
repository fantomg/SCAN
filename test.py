import pandas as pd
import snap
import random
from sklearn.cluster import MiniBatchKMeans
import joblib


test_data = pd.read_csv('test.csv', header=None)
node_user_data = pd.read_csv('node_user.csv', header=None)
node_package_data = pd.read_csv('node_package.csv', header=None)

# 加载模型
kmeans_user = joblib.load('mini_batch_kmeans_user_model.pkl')
kmeans_package = joblib.load('mini_batch_kmeans_package_model.pkl')

# 预测测试集
test_data['pred_user_id'] = test_data[0].apply(lambda x: kmeans_user.predict([node_user_data.iloc[x-1, 1:].values])[0])
test_data['pred_package_id'] = test_data[1].apply(lambda x: kmeans_package.predict([node_package_data.iloc[x-1, 1:].values])[0])

# 判断是否为一类套餐
test_data['is_class1'] = test_data.apply(lambda row: 1 if row['pred_package_id'] == row['pred_user_id'] else 0, axis=1)

# 计算正确率
accuracy = test_data['is_class1'].sum() / len(test_data)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

import random

# 随机选择十条数据
random_data = test_data.sample(n=10)

# 输出用户ID、预测套餐类别和真实套餐类别
for _, row in random_data.iterrows():
    user_id = row[0]
    pred_package_id = row['pred_package_id']
    true_package_id = row[1]
    print('User ID: {}, Predicted Package ID: {}, True Package ID: {}'.format(user_id, pred_package_id, true_package_id))