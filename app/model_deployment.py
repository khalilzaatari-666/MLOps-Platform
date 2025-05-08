from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv

load_dotenv()

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REMOTE_FOLDER = os.getenv("REMOTE_FOLDER")

client = Minio(
    endpoint = CONNECTION_STRING,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=True
)

def deploy_model_to_minio(destination_file: str, source_file: str) -> None:
    """
    Deploys a model to a MinIO bucket.

    Args:
        model_name (str): The name of the model to be deployed.
        model_path (str): The local path to the model file.
        bucket_name (str): The name of the MinIO bucket where the model will be stored.

    Raises:
        S3Error: If there is an error while uploading the model to MinIO.
    """
    try:
        # Check if the bucket exists, if not create it
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            print(f"Bucket {BUCKET_NAME} created successfully.")
        else:
            print(f"Bucket {BUCKET_NAME} already exists.")
        
        print(f"Uploading {source_file} to {destination_file}...")
        client.fput_object(BUCKET_NAME, destination_file, source_file)
        print(f"Model in {source_file} deployed successfully as {destination_file} to {BUCKET_NAME}.")
        return {
            "status": "SUCCESS",
            "model_name": destination_file,
            "model_path": source_file,
            "bucket_name": BUCKET_NAME
        }
    except S3Error as e:
        print(f"Error deploying model {source_file}: {e}")




