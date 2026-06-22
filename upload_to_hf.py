import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

def upload_dataset(file_path, repo_id, token):
    login(token=token)
    api = HfApi()
    
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f"Uploaded {file_path} to Hugging Face https://huggingface.co/datasets/{repo_id}")

if __name__ == '__main__':
    HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN", "invalid_token")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE_PATH = os.path.join(script_dir, "data", "input.txt")
    
    if HF_WRITE_TOKEN == "invalid_token":
        print("Please update HF_WRITE_TOKEN in the .env file before running!")
    else:
        upload_dataset(INPUT_FILE_PATH, "joshuajwoo/llm_data", HF_WRITE_TOKEN)