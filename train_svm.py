from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SVMClassifier") \
    .getOrCreate()

# GitHub raw URLs for datasets
train_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/TrainingDataset.csv"
val_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/ValidationDataset.csv"

# Load datasets directly from GitHub
print("Loading training dataset...")
train_data = spark.read.csv(train_url, header=True, inferSchema=True)

print("Loading validation dataset...")
val_data = spark.read.csv(val_url, header=True, inferSchema=True)

# Verify data loading
print("Training dataset schema:")
train_data.printSchema()

print("Validation dataset schema:")
val_data.printSchema()

# TODO: Add your ML training and validation logic here
# Example: Print row counts for verification
print(f"Training data row count: {train_data.count()}")
print(f"Validation data row count: {val_data.count()}")

# Stop Spark session
spark.stop()
