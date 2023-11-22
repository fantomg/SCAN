import pandas as pd
import joblib
from sklearn.cluster import KMeans

# 读取测试集数据
test_data = pd.read_csv('test.csv', header=None)

# 随机取样1000条数据
test_data = test_data.sample(n=1000, random_state=72)

# 加载训练好的模型
model = joblib.load('model.pkl')

# 提取测试集特征
X_test = test_data.iloc[:, 1:].values

# 使用模型进行预测
predicted_ids = model.predict(X_test)

# 将预测结果存储在新列中
test_data['predicted_id'] = predicted_ids

# 保存包含预测结果的文件
test_data.to_csv('predicted_test.csv', index=False, header=None)

# 读取节点文件
node_user_data = pd.read_csv('node_user.csv')

# 聚类分析
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(node_user_data.iloc[:, 0].values.reshape(-1, 1))

# 获取预测结果和真实ID的聚类标签
predicted_labels = kmeans.predict(test_data['predicted_id'].values.reshape(-1, 1))
true_labels = kmeans.predict(test_data.iloc[:, 0].values.reshape(-1, 1))

# 判断预测准确性
correct_predictions = 0
for i in range(len(predicted_labels)):
    if predicted_labels[i] == true_labels[i]:
        correct_predictions += 1

# 打印预测准确率
accuracy = correct_predictions / len(predicted_labels)
print("预测准确率：", accuracy)