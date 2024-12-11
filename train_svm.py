import urllib.request
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("SVMClassifier").getOrCreate()

# GitHub raw URLs for datasets
train_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/TrainingDataset.csv"
val_url = "https://raw.githubusercontent.com/SaiChowdaryBodapati/amazonwebserviceassignment2/main/ValidationDataset.csv"

# Download files locally
train_file = "/tmp/TrainingDataset.csv"
val_file = "/tmp/ValidationDataset.csv"
urllib.request.urlretrieve(train_url, train_file)
urllib.request.urlretrieve(val_url, val_file)

# Load datasets using Spark
print("Loading training dataset...")
train_data = spark.read.csv(train_file, header=True, inferSchema=True)

print("Loading validation dataset...")
val_data = spark.read.csv(val_file, header=True, inferSchema=True)

# Display schema
train_data.printSchema()
val_data.printSchema()

# Stop Spark session
spark.stop()
