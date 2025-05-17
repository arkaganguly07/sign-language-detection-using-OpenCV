import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Convert to NumPy array and ensure uniformity
data_list = data_dict['data']
labels = np.array(data_dict['labels'])

# Find the maximum feature length
max_length = max(len(sample) if isinstance(sample, (list, np.ndarray)) else 0 for sample in data_list)

# Pad or truncate sequences
data = np.array([
    np.pad(sample, (0, max_length - len(sample))) if len(sample) < max_length else sample[:max_length] 
    for sample in data_list
], dtype=np.float32)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train Random Forest with tuned parameters
model = RandomForestClassifier(
    n_estimators=500,  
    max_depth=None,  
    min_samples_split=2,  
    min_samples_leaf=1,  
    max_features="sqrt",  
    bootstrap=False,  
    random_state=42
)
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
