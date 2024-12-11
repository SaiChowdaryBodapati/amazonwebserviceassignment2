# Initialize Spark session
spark = SparkSession.builder.appName("SVMClassifier").getOrCreate()

# Replace 'your-bucket-name' with your actual bucket name
s3_path = "s3a://svmparallel/TrainingDataset.csv"

# Load the dataset from S3
data = spark.read.csv(s3_path, header=True, inferSchema=True)

# Prepare features and labels
feature_columns = data.columns[:-1]  # Exclude 'quality' column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# Split data into training and validation sets
train, val = data.randomSplit([0.8, 0.2], seed=42)


# Define the base SVM classifier
svm = LinearSVC(maxIter=100, regParam=0.1, labelCol="quality", featuresCol="features")

# Use One-vs-Rest for multi-class classification
ovr = OneVsRest(classifier=svm)

# Train the One-vs-Rest model
print("Training SVM classifier...")
ovr_model = ovr.fit(train)

# Perform predictions on the validation dataset
print("Evaluating the model...")
predictions = ovr_model.transform(val)

# Evaluate the model performance using F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score}")