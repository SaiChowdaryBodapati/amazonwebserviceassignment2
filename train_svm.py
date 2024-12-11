from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("SVMClassifier").getOrCreate()

# Define S3 bucket details
s3_bucket = "s3a://svmparallel/TrainingDataset.csv"

# Load data from S3
print("Loading training dataset from S3...")
try:
    data = spark.read.csv(s3_bucket, header=True, inferSchema=True)
except Exception as e:
    print(f"Error reading dataset from S3: {e}")
    exit()

# Convert all columns except "quality" to numeric
for column in data.columns:
    if column != "quality":
        data = data.withColumn(column, col(column).cast("double"))

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
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")

f1_score = evaluator_f1.evaluate(predictions)
accuracy = evaluator_accuracy.evaluate(predictions)

print(f"F1 Score: {f1_score}")
print(f"Accuracy: {accuracy}")
