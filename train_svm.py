from pyspark.sql import SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize SparkSession
spark = SparkSession.builder.appName("SVMClassifier").getOrCreate()

# Define S3 bucket details
s3_bucket = "s3a://svmparallel/TrainingDataset.csv"

# Load data from S3
print("Loading training dataset from S3...")
data = spark.read.csv(s3_bucket, header=True, inferSchema=True)

# Feature engineering
feature_columns = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data).select("features", "quality")

# Train-test split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Train SVM model
print("Training the SVM model...")
svm = LinearSVC(featuresCol="features", labelCol="quality", maxIter=10, regParam=0.1)
svm_model = svm.fit(train_data)

# Test the model
print("Evaluating the model...")
predictions = svm_model.transform(test_data)

# Evaluate performance
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

print(f"F1 Score: {f1_score}")

# Stop Spark session
spark.stop()
