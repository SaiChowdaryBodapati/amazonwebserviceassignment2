import boto3
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# S3 bucket and model details
bucket_name = "svmclassifier"  # Your S3 bucket name
model_file = "svm_model.pkl"   # Model file name in S3

# Step 1: Download the model from S3
print(f"Downloading model '{model_file}' from S3 bucket '{bucket_name}'...")
s3 = boto3.client('s3')

try:
    # Download the model file from the S3 bucket
    s3.download_file(bucket_name, model_file, model_file)
    print("Model downloaded successfully.")
except Exception as e:
    print(f"Failed to download model from S3: {e}")
    exit(1)

# Step 2: Load the trained model
print("Loading the model...")
try:
    with open(model_file, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load the model: {e}")
    exit(1)

# Step 3: Load the validation dataset
val_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/ValidationDataset.csv"
try:
    val_data = pd.read_csv(val_url, sep=';')
    print("Validation dataset loaded successfully.")
except Exception as e:
    print(f"Failed to load the validation dataset: {e}")
    exit(1)

# Step 4: Prepare the validation data
X_val = val_data.drop(columns=['quality'])
y_val = val_data['quality']

# Step 5: Make predictions
print("Making predictions on the validation dataset...")
try:
    y_pred = model.predict(X_val)

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
except Exception as e:
    print(f"Failed to make predictions: {e}")
    exit(1)
