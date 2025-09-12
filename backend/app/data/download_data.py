import os
import gdown
import requests
from bs4 import BeautifulSoup

# --- Settings ---
FOLDER_URL = "https://drive.google.com/drive/folders/176Y2RedBqGd063bEJq0uJmb7Fsfr-ehl"
OUTPUT_DIR = "./backend/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Get local file names ---
local_files = set(os.listdir(OUTPUT_DIR))

# --- Scrape shared folder page ---
print("🔍 Fetching file list from Google Drive folder...")
response = requests.get(FOLDER_URL)
soup = BeautifulSoup(response.text, "html.parser")

# --- Extract file IDs and names ---
# gdrive folder public pages have links with "/uc?id=<file_id>"
drive_files = {}
for a in soup.find_all("a"):
    href = a.get("href", "")
    if "uc?id=" in href:
        file_id = href.split("uc?id=")[1].split("&")[0]
        file_name = a.text.strip()
        drive_files[file_id] = file_name

print(f"Found {len(drive_files)} files in the Drive folder.\n")

# --- Download missing files only ---
for file_id, file_name in drive_files.items():
    local_path = os.path.join(OUTPUT_DIR, file_name)
    if file_name in local_files:
        print(f"✅ Skipping {file_name}, already exists.")
        continue

    print(f"⬇️ Downloading {file_name}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)

print("\n✅ Sync complete!")
