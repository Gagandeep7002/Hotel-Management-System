# train_classifier.py

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your data
data = np.load('data.py')  # Replace with your data loading logic
labels = np.load('labels.py')  # Replace with your labels loading logic

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier()  # Replace with your model
model.fit(X_train_scaled, y_train)

# Save the model, scaler, and class labels
model_dict = {
    'model': model,
    'scaler': scaler,
    'class_labels': {i: chr(ord('A') + i) for i in range(26)}
}
with open('./model.p', 'wb') as f:
    pickle.dump(model_dict, f)
