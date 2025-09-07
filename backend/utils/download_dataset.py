import gdown
import os

# Google Drive file ID
file_id = '1aby-qlnagD3KD0ZZ1LB6zbbIeNHyAlVC'
url = f'https://drive.google.com/uc?id={file_id}'

# Local path to save dataset
output_path = 'datasets/emails.csv'

# Create datasets folder if it doesn't exist
os.makedirs('datasets', exist_ok=True)

# Download if not exists
if not os.path.exists(output_path):
    print("Downloading dataset...")
    gdown.download(url, output_path, quiet=False)
    print("Download complete!")
else:
    print("Dataset already exists locally.")
