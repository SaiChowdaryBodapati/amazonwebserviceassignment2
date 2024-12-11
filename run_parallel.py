import boto3
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from multiprocessing import Pool, cpu_count
import os
import time

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
X_val = val_data.drop(columns=['quality']).values
y_val = val_data['quality'].values

# Function to make predictions on a chunk of data
def predict_chunk(indices):
    print(f"Worker {os.getpid()} processing indices: {indices}")
    start = time.time()
    chunk_X = X_val[indices]
    chunk_y = y_val[indices]
    predictions = model.predict(chunk_X)
    print(f"Worker {os.getpid()} finished in {time.time() - start:.2f} seconds")
    return predictions, chunk_y

# Step 5: Split the data for parallel processing
num_workers = cpu_count()  # Number of available CPUs
indices = np.array_split(np.arange(len(X_val)), num_workers)

# Step 6: Parallel processing
print("Making predictions in parallel...")
start_time = time.time()
with Pool(num_workers) as pool:
    results = pool.map(predict_chunk, indices)
end_time = time.time()

print(f"Parallel processing time: {end_time - start_time:.2f} seconds")

# Combine predictions and true labels
all_predictions = np.concatenate([result[0] for result in results])
all_labels = np.concatenate([result[1] for result in results])

# Step 7: Evaluate performance
print("\nEvaluating performance...")
accuracy = accuracy_score(all_labels, all_predictions)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions))
