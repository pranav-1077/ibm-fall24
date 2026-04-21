"""
Step 4: Filter satellite-street view pairs by distance, then augment BLIP captions
        using GPT-4o-mini and the corresponding street view images from S3.

Prerequisites:
    pip install openai boto3 pandas pillow tqdm
    Set OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION env vars.
"""

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import boto3
import openai
import pandas as pd
from botocore.exceptions import ClientError
from PIL import Image
from tqdm import tqdm

import config

openai.api_key = config.OPENAI_API_KEY

AUGMENTATION_PROMPT = """\
You are an expert in enhancing geospatial data captions.

Context:
- A remote sensing image has been analyzed.
- A street view image taken {distance} meters away from the remote sensing image location is provided.
- The following caption describes the scene in the remote sensing image: "{description}"

Task:
- Augment the provided caption by incorporating relevant contextual details observed in the provided street view image if possible
- Focus on objective features that enhance the spatial understanding of the scene (e.g., building types, road structures, vegetation presence, or land use).
- Avoid mentioning type of imagery (e.g., street view, remote sensing),
- Avoid mentioning details about weather, vehicles, or background
- Avoid using subjective language such as "lush" or "beautiful"

Example:
Input: "Some buildings and a field."
Output: "Multiple homes next to a paved road and a small agricultural field."

Limit output to 20 words. Use very basic language.\
"""


def augment_text(image, distance, description):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = AUGMENTATION_PROMPT.format(distance=distance, description=description)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        max_tokens=30,
        temperature=0,
    )
    return response.choices[0].message.content


def process_images_parallel(bucket_name, subfolder, input_df, output_csv, max_workers=None):
    if max_workers is None:
        max_workers = config.TEXT_AUGMENTATION_WORKERS

    s3 = boto3.client("s3")
    image_keys = []
    continuation_token = None

    print(f"Listing street view images in s3://{bucket_name}/{subfolder} ...")
    try:
        while True:
            kwargs = {"Bucket": bucket_name, "Prefix": subfolder}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = s3.list_objects_v2(**kwargs)
            if "Contents" not in resp:
                print("No images found.")
                break
            image_keys.extend(k["Key"] for k in resp["Contents"] if k["Key"].endswith(".jpg"))
            if resp.get("IsTruncated"):
                continuation_token = resp["NextContinuationToken"]
            else:
                break
    except ClientError as e:
        print(f"Error accessing bucket: {e}")
        return

    print(f"Found {len(image_keys)} street view images.")

    def process_row(row):
        image_id = row["image_id"]
        match = next((k for k in image_keys if image_id in k), None)
        if not match:
            return {"image_id": image_id, "description": "No matching image"}
        try:
            img_data = s3.get_object(Bucket=bucket_name, Key=match)["Body"].read()
            with Image.open(BytesIO(img_data)) as image:
                image = image.convert("RGB")
                augmented = augment_text(image, row["distance"], row["description"])
                return {"image_id": image_id, "description": augmented.strip('"')}
        except Exception as e:
            print(f"Error on {match}: {e}")
            return {"image_id": image_id, "description": f"Error: {e}"}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row): row for _, row in input_df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting captions"):
            results.append(future.result())

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    df = pd.read_csv(config.JOINED_CSV)
    df = df.sort_values("distance", ascending=True)
    filtered = df[df["distance"] <= config.MAX_DISTANCE_FILTER]
    print(f"Filtered to {len(filtered)} samples (distance <= {config.MAX_DISTANCE_FILTER} m)")

    process_images_parallel(
        bucket_name=config.S3_BUCKET,
        subfolder=config.S3_STREET_SUBFOLDER,
        input_df=filtered,
        output_csv=config.AUGMENTED_CSV,
    )
