import os
from huggingface_hub import HfApi, login

def upload_dataset(file_path, repo_id, token):
    """
    Uploads a local file to a Hugging Face dataset repository.
    
    Args:
        file_path (str): The local path to the file you want to upload.
        repo_id (str): Your HF repository ID, e.g., "joshuajwoo/my-web-dataset"
        token (str): Your Hugging Face write token.
    """
    # 1. Login with the provided write token
    login(token=token)
    
    # 2. Initialize the API
    api = HfApi()
    
    # 3. Create the repository if it doesn't exist (type="dataset")
    print(f"Ensuring repository '{repo_id}' exists...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    # 4. Upload the file
    print(f"Uploading {file_path} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f"Successfully uploaded {file_path} to Hugging Face! (https://huggingface.co/datasets/{repo_id})")

if __name__ == '__main__':
    # --- CONFIGURATION ---
    # Replace these with your actual details
    HF_WRITE_TOKEN = "hf_YOUR_WRITE_TOKEN_HERE"  # Keep this secret!
    HF_REPO_ID = "your-username/your-dataset-name" # e.g., "joshuajwoo/my-scraped-data"
    
    # Path to the data we just scraped
    script_dir = os.path.dirname(os.path.abspath(__file__))
    LOCAL_FILE_PATH = os.path.join(script_dir, "data", "input.txt")
    
    # Run the upload
    if HF_WRITE_TOKEN.startswith("hf_YOUR"):
        print("Please update HF_WRITE_TOKEN and HF_REPO_ID in the script before running!")
    else:
        upload_dataset(LOCAL_FILE_PATH, HF_REPO_ID, HF_WRITE_TOKEN)
