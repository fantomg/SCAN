# kmeans.py
import joblib
import pandas as pd
from sklearn.cluster import KMeans

# Read the node-user data
node_user_data = pd.read_csv('node_user.csv')

# Perform clustering analysis
kmeans = KMeans(n_clusters=10, random_state=43, n_init=10)
kmeans.fit(node_user_data.iloc[:, 0].values.reshape(-1, 1))

# Save the trained K-means model
joblib.dump(kmeans, 'kmeans_model.pkl')
