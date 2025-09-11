import gdown
import os

# Google Drive file ID
file_id = "1rPRHQzE8U3Vox2JSuIK6Bo2pgEhk86lb"

# Output path
output_path = "backend/app/data/Final_Emails.csv"

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
