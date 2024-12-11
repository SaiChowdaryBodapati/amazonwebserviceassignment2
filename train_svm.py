from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Initialize Spark session
print("Initializing Spark session...")
spark = SparkSession.builder.appName("SVMClassifier").getOrCreate()

# Define S3 bucket details
s3_bucket = "s3a://svmparallel/TrainingDataset.csv"

# Load data from S3
print("Loading training dataset from S3...")
data = spark.read.csv(s3_bucket, header=True, inferSchema=True)
print("Data loaded successfully. Schema:")
data.printSchema()

# Debug: Show a sample of the data
print("Sample data:")
data.show(5)

# Ensure all features are numeric
print("Casting all feature columns to numeric...")
for column in data.columns:
    if column != "quality":  # Assuming 'quality' is the label column
        data = data.withColumn(column, col(column).cast("float"))
print("Casting completed. Updated schema:")
data.printSchema()

# Feature engineering
print("Preparing features for model training...")
feature_columns = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "quality")
print("Feature engineering completed. Sample transformed data:")
data.show(5)

# Train-test split
print("Splitting data into train and test sets...")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)
print(f"Data split completed. Training data count: {train_data.count()}, Test data count: {test_data.count()}")

# Train SVM model
print("Training the SVM model...")
svm = LinearSVC(featuresCol="features", labelCol="quality", maxIter=10, regParam=0.1)
svm_model = svm.fit(train_data)
print("Model training completed.")

# Test the model
print("Generating predictions on the test set...")
predictions = svm_model.transform(test_data)
print("Predictions generated. Sample predictions:")
predictions.select("features", "quality", "prediction").show(5)

# Evaluate performance
print("Evaluating model performance...")
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Stop Spark session
print("Stopping Spark session...")
spark.stop()
print("Spark session stopped.")
