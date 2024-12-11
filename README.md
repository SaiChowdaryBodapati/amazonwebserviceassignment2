# Repository: amazonwebserviceassignment2

## Overview
This repository contains the implementation of an SVM-based machine learning application designed to predict wine quality. The project leverages parallel processing and AWS services, and includes several scripts for different stages of the process, including training, validation, and Docker integration.

---

## Directory Structure
```
.
├── Dockerfile              # Docker configuration to containerize the application
├── README.md               # Documentation for repository usage
├── TrainingDataset.csv     # Training dataset for the SVM model
├── ValidationDataset.csv   # Validation dataset for model evaluation
├── docker_app.py           # Python script for executing the application in Docker
├── requirements.txt        # Dependencies for the project
├── run_parallel.py         # Parallelized implementation for predictions
├── svm_model.pkl           # Pre-trained SVM model stored as a pickle file
├── train_svm.py            # Script to train the SVM model
```

---

## Prerequisites

- Python 3.9 or higher
- Docker installed and configured
- Access to AWS S3 bucket (`svmclassifier`) containing the pre-trained model (`svm_model.pkl`)
- Necessary Python dependencies listed in `requirements.txt`

---

## Scripts and Execution

### **1. `train_svm.py`**
This script trains the SVM model using the `TrainingDataset.csv` file and saves the model as `svm_model.pkl`.

**Execution Command:**
```bash
python3 train_svm.py
```

### **2. `run_parallel.py`**
This script performs predictions on the `ValidationDataset.csv` using the pre-trained SVM model in parallel using Python's `multiprocessing` module.

**Execution Command:**
```bash
python3 run_parallel.py
```

**Output Includes:**
- Parallel execution logs showing worker IDs
- Validation accuracy
- Classification report

### **3. `docker_app.py`**
This script runs the application inside a Docker container, pulling the pre-trained model from AWS S3 and validating it against the `ValidationDataset.csv`.

**Steps to Run in Docker:**
1. **Build the Docker Image:**
   ```bash
   docker build -t saitejdeep/cluster_ml_application:latest .
   ```
2. **Run the Docker Container:**
   ```bash
   docker run --rm -it saitejdeep/cluster_ml_application:latest
   ```

### **4. Dockerfile**
The `Dockerfile` is used to containerize the application. It installs necessary dependencies, copies project files, and sets `docker_app.py` as the entry point.

**Build Command:**
```bash
sudo docker build -t saitejdeep/cluster_ml_application:latest .
```

**Run Command:**
```bash
sudo docker run --rm -it saitejdeep/cluster_ml_application:latest
```

### **5. `requirements.txt`**
This file lists all Python dependencies required by the project:
```
boto3
pandas
numpy
scikit-learn
```

**Install Dependencies:**
```bash
pip3 install -r requirements.txt
```

---

## AWS Integration
- **Model Storage:** The pre-trained SVM model (`svm_model.pkl`) is stored in the S3 bucket `svmclassifier`.
- **Accessing S3:** The `boto3` library is used to download the model during execution.

---

## Parallel Execution
`run_parallel.py` utilizes Python's `multiprocessing` module to divide the validation dataset into chunks and process them in parallel.

**Key Highlights:**
- The script automatically detects the number of available CPU cores.
- Each worker processes a chunk of data independently, ensuring efficient execution.

**Sample Output:**
```
Worker 12345 processing indices: [0:39]
Worker 12346 processing indices: [40:79]
Worker 12347 processing indices: [80:119]
Worker 12348 processing indices: [120:159]
Parallel processing time: 0.03 seconds
Validation Accuracy: 56.88%
Classification Report:
...
```

---

## Notes
1. Ensure proper permissions for accessing AWS resources.
2. If running inside Docker, the `requirements.txt` file must be up-to-date with all dependencies.
3. For data imbalance issues, consider retraining the model with additional data or balancing techniques.

---

## Troubleshooting
- **Feature Name Warning:** Ensure feature names are consistent when predicting.
- **UndefinedMetricWarning:** Use `zero_division=0` in the classification report to handle undefined metrics for labels with no predicted samples.
- **Permission Denied Errors in Docker:** Use `sudo` or add your user to the `docker` group:
  ```bash
  sudo usermod -aG docker $USER
  ```

---

## Future Enhancements
- Integrate more robust logging for better debugging.
- Add support for real-time predictions via an API (e.g., Flask or FastAPI).
- Automate training and validation with CI/CD pipelines.
