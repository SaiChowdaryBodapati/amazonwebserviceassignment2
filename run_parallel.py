from concurrent.futures import ThreadPoolExecutor
import paramiko
import boto3

# List of slave node IPs
slave_ips = [
    "54.174.233.122",
    "18.212.125.214",
    "3.82.50.115",
    "52.91.180.181",
    "18.234.202.13",
    "18.207.200.245",
]

# Function to execute svm.py on a slave node
def run_command(slave_ip):
    user = "hadoop"  # SSH username
    remote_command = "python3 ~/amazonwebserviceassignment2/svm.py"  # Command to execute svm.py
    s3_bucket_name = "svmparallel"  # Your S3 bucket name
    s3_client = boto3.client('s3')

    try:
        # Establish an SSH connection to the slave
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=slave_ip, username=user)  # No private key required for password-less SSH

        # Execute the remote command
        print(f"Executing command on {slave_ip}...")
        stdin, stdout, stderr = client.exec_command(remote_command)

        # Wait for command to complete and fetch output
        print(f"Output from {slave_ip}:\n{stdout.read().decode()}")
        print(f"Error from {slave_ip}:\n{stderr.read().decode()}")

        # SCP to fetch the results from the slave
        print(f"Fetching results from {slave_ip}...")
        sftp = client.open_sftp()
        local_model_path = f"./{slave_ip}_svm_model.pkl"
        local_results_path = f"./{slave_ip}_validation_results.txt"
        sftp.get("/home/hadoop/svm_model.pkl", local_model_path)
        sftp.get("/home/hadoop/validation_results.txt", local_results_path)
        sftp.close()

        # Upload the results to S3
        print(f"Uploading results from {slave_ip} to S3...")
        s3_client.upload_file(local_model_path, s3_bucket_name, f"{slave_ip}/svm_model.pkl")
        s3_client.upload_file(local_results_path, s3_bucket_name, f"{slave_ip}/validation_results.txt")
        print(f"Results from {slave_ip} successfully uploaded to S3.")

        client.close()
    except Exception as e:
        print(f"Error connecting to {slave_ip}: {e}")

# Run the commands in parallel across all slave nodes
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(slave_ips)) as executor:
        executor.map(run_command, slave_ips)
