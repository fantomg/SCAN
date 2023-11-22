import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import joblib

# Read the training dataset
train_data = pd.read_csv('test.csv', header=None)

# Randomly sample 30,000 data points
train_data = train_data.sample(n=30000, random_state=42)

# Split features and labels
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

# Create Gaussian Naive Bayes classifier model
model = GaussianNB()

# Fit the model using the first call to partial_fit with classes parameter
model.partial_fit(X, y, classes=np.unique(y))

# Save the model
joblib.dump(model, 'model.pkl')