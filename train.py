import pandas as pd
import snap
import pickle
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

# 读取数据集
data = pd.read_csv('train.csv', header=None)

# 创建图网络
graph = snap.TUNGraph.New()
for row in tqdm(data.iterrows(), total=len(data), desc='读取数据'):
    srcID, dstID, attr1, attr2 = row[1][0], row[1][1], row[1][2], row[1][3]
    if not graph.IsNode(int(srcID)):
        graph.AddNode(int(srcID))
    if not graph.IsNode(int(dstID)):
        graph.AddNode(int(dstID))
    graph.AddEdge(int(srcID), int(dstID))

# 保存图模型到文件
FOut = snap.TFOut("graph_model.bin")
graph.Save(FOut)
FOut.Flush()

# 使用Snap库中的方法进行用户聚类
user_clusters = snap.TCnComV()
snap.GetWccs(graph, user_clusters)

# 将用户聚类结果作为新的特征添加到数据集中
user_cluster_dict = {i: user_clusters[i].Len() for i in range(len(user_clusters))}
data['user_cluster'] = data[0].map(user_cluster_dict)

# 提取特征和目标变量
X = data[[0, 1]].astype(str)
y = data[2]

# 使用SVD进行用户套餐推荐系统
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

# 保存SVD模型到文件
with open('svd_model.pkl', 'wb') as file:
    pickle.dump(svd, file)