import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # ✅ Add this
import pickle

# Load dataset
data = pd.read_csv('heart.csv')

X = data.drop('target', axis=1)
y = data['target']

# ✅ Scale features for better convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# ✅ Save both model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

print("Model trained and saved successfully.")




