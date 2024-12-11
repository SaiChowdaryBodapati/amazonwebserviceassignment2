import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import boto3
import os

# Load the datasets
train_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/TrainingDataset.csv"
val_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/ValidationDataset.csv"

print("Loading datasets...")
train_data = pd.read_csv(train_url, sep=';')
val_data = pd.read_csv(val_url, sep=';')

# Split features (X) and target (y) for training and validation
X_train = train_data.drop(columns=['quality'])
y_train = train_data['quality']
X_val = val_data.drop(columns=['quality'])
y_val = val_data['quality']

# Define the SVM Classifier with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],              # Regularization parameter
    'kernel': ['linear', 'rbf'],    # Kernel types
    'gamma': ['scale', 'auto']      # Kernel coefficient
}

svm_model = SVC()

print("Setting up Grid Search...")
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1  # Use all available cores
)

print("Performing Grid Search...")
grid_search.fit(X_train, y_train)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Train the best model on the full training data
print("Training the best model on the full training data...")
best_model.fit(X_train, y_train)

# Save the trained model to a file
model_file = "svm_model.pkl"
with open(model_file, "wb") as file:
    pickle.dump(best_model, file)
print(f"Model saved locally as '{model_file}'.")

# Upload the model to S3
s3 = boto3.client('s3')
bucket_name = "svmparallel"  # Your S3 bucket name
try:
    s3.upload_file(model_file, bucket_name, model_file)
    print(f"Model uploaded to S3 bucket '{bucket_name}' as '{model_file}'.")
except Exception as e:
    print(f"Failed to upload model to S3: {e}")

# Validate the model
print("Validating the model...")
y_val_pred = best_model.predict(X_val)

# Calculate accuracy and display classification report
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Save validation results to a file
results_file = "validation_results.txt"
with open(results_file, "w") as file:
    file.write(f"Validation Accuracy: {val_accuracy * 100:.2f}%\n")
    file.write("\nValidation Classification Report:\n")
    file.write(classification_report(y_val, y_val_pred))
print(f"Validation results saved locally as '{results_file}'.")

# Upload validation results to S3
try:
    s3.upload_file(results_file, bucket_name, results_file)
    print(f"Validation results uploaded to S3 bucket '{bucket_name}' as '{results_file}'.")
except Exception as e:
    print(f"Failed to upload validation results to S3: {e}")
