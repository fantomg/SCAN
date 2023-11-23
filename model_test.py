import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Read the test data
test_data = pd.read_csv('test.csv', header=None)
test_data = test_data.sample(n=1000, random_state=72)

# Load the trained model
model = joblib.load('model.pkl')

# Extract features from the test data
X_test = test_data.iloc[:, 1:].values

# Make predictions using the model
predicted_ids = model.predict(X_test)

# Store the predicted IDs in a new column
test_data['predicted_id'] = predicted_ids

# Save the file with predicted results
test_data.to_csv('predicted_test.csv', index=False, header=None)

# Load the K-means model
kmeans = joblib.load('kmeans_model.pkl')

# Get the clustering labels for predicted IDs and true IDs
predicted_labels = kmeans.predict(test_data['predicted_id'].values.reshape(-1, 1))
true_labels = kmeans.predict(test_data.iloc[:, 0].values.reshape(-1, 1))

# Calculate the prediction accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Calculate precision and recall
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')

# Print the performance metrics
print("Prediction Accuracy: {:.2f}%".format(accuracy * 100))
print("F1 Score: {:.2f}".format(f1))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# Create a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Visualize the prediction results in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_data.iloc[:, 0], test_data['predicted_id'], c=predicted_labels)
ax.set_xlabel('True ID')
ax.set_ylabel('Predicted ID')
ax.set_zlabel('Cluster Label')
ax.set_title('Prediction Results')
plt.show()
