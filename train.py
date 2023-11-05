import numpy as np
import pandas as pd
import snap
import random
from sklearn.cluster import MiniBatchKMeans
import joblib

# 读取训练数据
train_data = pd.read_csv('train.csv', header=None)
node_user_data = pd.read_csv('node_user.csv', header=None)
node_package_data = pd.read_csv('node_package.csv', header=None)

# 将数据分成301份
num_parts = 301
train_data_parts = np.array_split(train_data, num_parts)

# 构建图网络
G = snap.TNEANet.New()

for part in train_data_parts:
    part.reset_index(drop=True, inplace=True)  # 重置索引
    for i in range(len(part)):
        user_id = int(part.loc[i, 0])
        package_id = int(part.loc[i, 1])
        attr1 = part.loc[i, 2]
        attr2 = part.loc[i, 3]

        if not G.IsNode(user_id):
            G.AddNode(user_id)
        if not G.IsNode(package_id):
            G.AddNode(package_id)
        G.AddEdge(user_id, package_id)
        G.AddFltAttrDatE(G.GetEI(user_id, package_id), attr1, "Attr1")
        G.AddFltAttrDatE(G.GetEI(user_id, package_id), attr2, "Attr2")

# 获取节点属性值
node_user_attrs = node_user_data.iloc[:, 1:].values
node_package_attrs = node_package_data.iloc[:, 1:].values

# 使用Mini-Batch K-means算法进行训练
kmeans_user = MiniBatchKMeans(n_clusters=10, batch_size=1000, random_state=0).fit(node_user_attrs)
kmeans_package = MiniBatchKMeans(n_clusters=10, batch_size=1000, random_state=0).fit(node_package_attrs)

# 保存模型
joblib.dump(kmeans_user, 'mini_batch_kmeans_user_model.pkl')
joblib.dump(kmeans_package, 'mini_batch_kmeans_package_model.pkl')