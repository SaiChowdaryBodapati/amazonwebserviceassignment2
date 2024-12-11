from concurrent.futures import ThreadPoolExecutor
import paramiko

# List of EC2 instance details
instances = [
    {"host": "54.174.233.122", "user": "hadoop", "key_path": "C:\\Users\\saite\\Downloads\\newpair1.pem"},
    {"host": "18.212.125.214", "user": "hadoop", "key_path": "C:\\Users\\saite\\Downloads\\newpair1.pem"},
    {"host": "3.82.50.115", "user": "hadoop", "key_path": "C:\\Users\\saite\\Downloads\\newpair1.pem"},
    {"host": "52.91.180.181", "user": "hadoop", "key_path": "C:\\Users\\saite\\Downloads\\newpair1.pem"},
    {"host": "18.234.202.13", "user": "hadoop", "key_path": "C:\\Users\\saite\\Downloads\\newpair1.pem"},
    {"host": "18.207.200.245", "user": "hadoop", "key_path": "C:\\Users\\saite\\Downloads\\newpair1.pem"},
]

# Function to run a command on an EC2 instance
def run_command(instance):
    host = instance["host"]
    user = instance["user"]
    key_path = instance["key_path"]
    command = "echo $(hostname) is running a distributed task"  # Replace with your desired command

    try:
        # Establish an SSH connection
        key = paramiko.RSAKey.from_private_key_file(key_path)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, username=user, pkey=key)

        # Execute the command
        stdin, stdout, stderr = client.exec_command(command)
        print(f"Output from {host}:\n{stdout.read().decode()}")
        print(f"Error from {host}:\n{stderr.read().decode()}")

        client.close()
    except Exception as e:
        print(f"Error connecting to {host}: {e}")

# Run the commands in parallel
with ThreadPoolExecutor(max_workers=len(instances)) as executor:
    executor.map(run_command, instances)
