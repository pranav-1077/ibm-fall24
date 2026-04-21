"""
Step 3: Run BLIP captioning on satellite patches stored in S3.
        Writes (image_name, description) pairs to a CSV.

Prerequisites:
    pip install transformers torch torchvision pillow tqdm boto3
    Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION env vars.
"""

from io import BytesIO

import boto3
import pandas as pd
import torch
from botocore.exceptions import ClientError
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

import config


def build_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-to-text", model=config.BASE_BLIP_MODEL, device=device)


def run_inference(pipe, bucket_name, subfolder, output_csv):
    s3 = boto3.client("s3")
    all_keys = []
    continuation_token = None

    print(f"Listing .tif files in s3://{bucket_name}/{subfolder} ...")
    try:
        while True:
            kwargs = {"Bucket": bucket_name, "Prefix": subfolder}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = s3.list_objects_v2(**kwargs)

            if "Contents" not in resp:
                print("No images found.")
                break

            all_keys.extend(k["Key"] for k in resp["Contents"] if k["Key"].endswith(".tif"))

            if resp.get("IsTruncated"):
                continuation_token = resp["NextContinuationToken"]
            else:
                break
    except ClientError as e:
        print(f"Error accessing bucket: {e}")
        return

    print(f"Found {len(all_keys)} images.")
    results = []

    for key in tqdm(all_keys, desc="Running BLIP inference", unit="image"):
        try:
            img_data = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
            with Image.open(BytesIO(img_data)) as image:
                image = image.convert("RGB")
                caption = pipe(image)[0]["generated_text"]
                results.append({"image_name": key, "description": caption})
        except Exception as e:
            print(f"Error on {key}: {e}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    pipe = build_pipeline()
    run_inference(pipe, config.S3_BUCKET, config.S3_TIF_SUBFOLDER, config.DESCRIPTIONS_CSV)
