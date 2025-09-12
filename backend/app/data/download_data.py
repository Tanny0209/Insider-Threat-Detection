import os
import gdown

# --- Settings ---
FOLDER_URL = "https://drive.google.com/drive/folders/176Y2RedBqGd063bEJq0uJmb7Fsfr-ehl"
OUTPUT_DIR = "./backend/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- List files locally ---
local_files = set(os.listdir(OUTPUT_DIR))

# --- List of files with Google Drive IDs (can be obtained from shared folder) ---
drive_files = {
    "18c-bfSXd29xqFrtQouFqWbYwrcj_Iimb": "edges.pkl",
    "1lrrUv1geuP6wZ3QI56TUqZ1FP__FwVA6": "Emails.csv",
    "1O-3zOYrcylGwzXmWCK6scZByVet7wE_-": "node_mapping.pkl",
    "17SkkAlMYToyE1J7mlER6z_NlYimq3hKa": "pyg_graph_basic.pt",
    "1GQcCVcmkLOn7cBNItE6CkvKQNPwrzCbF": "pyg_graph_full.pt",
    "1wCj9fO04j2k7QeRLRn1k9YRrMh8pIjm9": "pyg_graph.pt",
    "1cMzn2JUUgTPzj8zgtF7C3g07SJhlphpI": "top_attachment_types.pkl",
    "1kKilSBRcb5xHnm7FYqYxILhgKEEEDM2Z": "top_domains.pkl",
    "1mDiViFOQcjSZ9l7S8xADGltXgaLUUWSR": "user_graph.pkl",
}

# --- Download missing files only ---
for file_id, file_name in drive_files.items():
    local_path = os.path.join(OUTPUT_DIR, file_name)
    if file_name in local_files:
        print(f"✅ Skipping {file_name}, already exists.")
        continue

    print(f"⬇️ Downloading {file_name}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)

print("\n✅ Sync complete!")
