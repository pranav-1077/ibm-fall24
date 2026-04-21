"""
Step 2b: Upload local patch folders to S3.

Prerequisites:
    pip install boto3
    Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION env vars.
"""

import os

import boto3
from botocore.exceptions import NoCredentialsError

import config


def upload_folder_to_s3(local_folder, bucket_name, s3_subfolder):
    s3 = boto3.client("s3")
    for filename in os.listdir(local_folder):
        file_path = os.path.join(local_folder, filename)
        if not os.path.isfile(file_path):
            continue
        s3_key = f"{s3_subfolder}/{filename}"
        try:
            s3.upload_file(file_path, bucket_name, s3_key)
            print(f"Uploaded {filename} -> s3://{bucket_name}/{s3_key}")
        except NoCredentialsError:
            print("AWS credentials not found. Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.")
            return
        except Exception as e:
            print(f"Error uploading {filename}: {e}")


if __name__ == "__main__":
    upload_folder_to_s3(config.TIFF_PATCHES_FOLDER, config.S3_BUCKET, config.S3_TIF_SUBFOLDER)
    upload_folder_to_s3(config.STREET_IMAGES_FOLDER, config.S3_BUCKET, config.S3_STREET_SUBFOLDER)
