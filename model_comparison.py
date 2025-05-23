import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

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

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Dictionary to store results
results = {
    'Model': [],
    'Accuracy': [],
    'Cross-Validation Score': []
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(x_train, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, data, labels, cv=5)
    cv_mean = cv_scores.mean()
    
    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Cross-Validation Score'].append(cv_mean)
    
    # Print detailed metrics
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation score: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()

# Create comparison DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(results_df.sort_values('Accuracy', ascending=False))

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Accuracy'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png')
plt.close()

# Plot cross-validation scores
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Cross-Validation Score'])
plt.title('Model Cross-Validation Score Comparison')
plt.xlabel('Models')
plt.ylabel('Cross-Validation Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_cv_score_comparison.png')
plt.close()

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False) 