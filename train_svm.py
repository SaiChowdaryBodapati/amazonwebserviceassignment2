import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Define GitHub raw dataset URL
github_url = "https://raw.githubusercontent.com/<your-username>/<your-repo>/<branch>/TrainingDataset.csv"

# Load data from GitHub
print("Loading training dataset from GitHub...")
response = requests.get(github_url)
data = pd.read_csv(pd.compat.StringIO(response.text))
print("Data loaded successfully. Sample data:")
print(data.head())

# Ensure all features are numeric
print("Casting all feature columns to numeric...")
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()
print("Casting completed. Updated data sample:")
print(data.head())

# Feature engineering
print("Preparing features for model training...")
X = data.drop(columns=['quality'])  # Assuming 'quality' is the label column
y = data['quality']

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(f"Data split completed. Training data size: {len(X_train)}, Test data size: {len(X_test)}")

# Train SVM model
print("Training the SVM model...")
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
print("Model training completed.")

# Test the model
print("Generating predictions on the test set...")
y_pred = svm.predict(X_test)
print("Predictions generated. Sample predictions:")
print(y_pred[:5])

# Evaluate performance
print("Evaluating model performance...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
