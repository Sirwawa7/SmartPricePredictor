"""
AI Project: House Price Prediction
This script demonstrates the use of various machine learning algorithms
for predicting house prices using a simulated dataset.
Algorithms: Linear Regression, K-Means Clustering, Decision Tree, Neural Network.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Simulated Dataset
num_samples = 1000
X = pd.DataFrame({
    'square_footage': np.random.randint(500, 3500, num_samples),
    'num_bedrooms': np.random.randint(1, 6, num_samples),
    'age_of_house': np.random.randint(0, 50, num_samples),
    'proximity_to_city': np.random.uniform(0, 1, num_samples)
})
y = (
    X['square_footage'] * 0.3 +
    X['num_bedrooms'] * 50 -
    X['age_of_house'] * 2 +
    X['proximity_to_city'] * 200 +
    np.random.normal(0, 50, num_samples)
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lr = lin_reg.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# 2. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(X_train_scaled)

# 3. Decision Tree
price_categories = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, price_categories, test_size=0.2, random_state=RANDOM_STATE
)
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(X_train_cls, y_train_cls)
y_pred_tree = tree.predict(X_test_cls)
accuracy_tree = accuracy_score(y_test_cls, y_pred_tree)

# 4. Neural Network
model = Sequential([
    Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
y_pred_nn = model.predict(X_test_scaled).flatten()
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

# Results
print("Linear Regression MSE:", mse_lr, "| R²:", r2_lr)
print("K-Means Clustering Labels (first 10):", clusters[:10])
print("Decision Tree Accuracy:", accuracy_tree)
print("Neural Network MSE:", mse_nn, "| R²:", r2_nn)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, label="Linear Regression")
plt.scatter(y_test, y_pred_nn, alpha=0.6, label="Neural Network")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Model Predictions vs. Actual Values")
plt.legend()
plt.show()
