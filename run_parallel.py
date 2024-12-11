from concurrent.futures import ThreadPoolExecutor
import paramiko

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

    try:
        # Establish an SSH connection to the slave
        print(f"Connecting to {slave_ip}...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=slave_ip, username=user)

        # Execute the remote command
        print(f"Executing SVM script on {slave_ip}...")
        stdin, stdout, stderr = client.exec_command(remote_command)

        # Wait for command to complete and fetch output
        output = stdout.read().decode()
        error = stderr.read().decode()

        if output:
            print(f"Output from {slave_ip}:\n{output}")
        if error:
            print(f"Error from {slave_ip}:\n{error}")

        client.close()
    except Exception as e:
        print(f"Error connecting to {slave_ip}: {e}")

# Run the commands in parallel across all slave nodes
if __name__ == "__main__":
    print("Starting parallel execution of SVM scripts...")
    with ThreadPoolExecutor(max_workers=len(slave_ips)) as executor:
        executor.map(run_command, slave_ips)
    print("Parallel execution completed.")
