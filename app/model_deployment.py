from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv
from datetime import timedelta
import requests

load_dotenv()

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REMOTE_FOLDER = os.getenv("REMOTE_FOLDER")

EXPIRATION = timedelta(hours=1)

def deploy_model_to_minio(source_file: str, destination_file: str) -> None:
    """
    Deploys a model to a MinIO bucket.

    Args:
        model_name (str): The name of the model to be deployed.
        model_path (str): The local path to the model file.
        bucket_name (str): The name of the MinIO bucket where the model will be stored.

    Raises:
        S3Error: If there is an error while uploading the model to MinIO.
    """
    # Check if the bucket exists, if not create it
    client = Minio(
        endpoint = CONNECTION_STRING,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=True
    )

    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Bucket {BUCKET_NAME} created successfully.")
    else:
        print(f"Bucket {BUCKET_NAME} already exists.")

    try: 
        presigned_url = client.presigned_put_object(BUCKET_NAME, destination_file, expires=EXPIRATION)
        print(f"Presigned URL: {presigned_url}")

        with open(source_file, "rb") as file:
            response = requests.put(presigned_url, data=file)
            if response.status_code == 200:
                print(f"Successfully uploaded {source_file} to {destination_file}")
                return {
                    "status": "success",
                    "source_file": source_file,
                    "destination_file": destination_file
                }
            else:
                print(f"Failed to upload file: {response.status_code}, {response.text}")
                return {
                    "status": "failed",
                    "message": response.text
                }

    except S3Error as e:
        print(f"Error deploying model {source_file}: {e}")




