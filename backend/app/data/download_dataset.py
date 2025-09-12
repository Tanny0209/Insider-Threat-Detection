import gdown
import os

# Google Drive file ID
file_id = "1lrrUv1geuP6wZ3QI56TUqZ1FP__FwVA6"

# Output path
output_path = "backend/app/data/Emails.csv"

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Check if file already exists
if os.path.exists(output_path):
    print(f"Dataset already exists at {output_path}. Skipping download.")
else:
    # Construct URL
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download file
    print("Downloading dataset...")
    gdown.download(url, output_path, quiet=False)
    print(f"Dataset downloaded to {output_path}")