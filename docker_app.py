import boto3
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# S3 bucket details
bucket_name = "svmclassifier"
model_file = "svm_model.pkl"

# Download the model from S3
print(f"Downloading model '{model_file}' from S3 bucket '{bucket_name}'...")
s3 = boto3.client('s3')
try:
    s3.download_file(bucket_name, model_file, model_file)
    print("Model downloaded successfully.")
except Exception as e:
    print(f"Failed to download model from S3: {e}")
    exit(1)

# Load the model
print("Loading the model...")
with open(model_file, "rb") as file:
    model = pickle.load(file)
print("Model loaded successfully.")

# Load the validation dataset
val_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/ValidationDataset.csv"
val_data = pd.read_csv(val_url, sep=';')

# Prepare the data
X_val = val_data.drop(columns=['quality'])
y_val = val_data['quality']

# Make predictions
print("Making predictions on the validation dataset...")
y_pred = model.predict(X_val)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
