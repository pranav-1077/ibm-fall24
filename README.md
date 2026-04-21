# Novel Synthetic Labelling System for Scalable Remote Sensing Captioning

A Fall 2024 [Data Discovery](https://cdss.berkeley.edu/discovery) project in partnership with **IBM**, advised by **Ranjan Sinha** and **Karina Kervin**.

**Poster:** [Environmental Intelligence Using Machine Learning](https://cdss.berkeley.edu/project/environmental-intelligence-using-machine-learning)

---

## Overview

Annotating satellite imagery at scale is expensive and time-consuming. This project introduces a synthetic labelling pipeline that eliminates manual annotation by aligning satellite image tiles with street-level imagery and using a vision-language model to generate high-quality captions automatically.

The generated dataset is then used to fine-tune a remote sensing captioning model (BLIP), and the improvement over the baseline is measured using RemoteCLIP cosine similarity.

---

## Pipeline

```
LandCover.AI TIFs
       │
       ▼
1. lcai_pipeline.py   — split orthophotos into 512×512 patches, geocode each patch,
                        fetch the nearest Google Street View image
       │
       ▼
2. blip_inference.py  — run BLIP on satellite patches to produce initial captions
       │
       ▼
3. s3_upload.py       — upload patches + street view images to S3
       │
       ▼
4. join_tables.py     — merge coordinate table with BLIP captions
       │
       ▼
5. text_augmentation.py — filter by street view distance, augment captions with
                          GPT-4o-mini using the paired street view image
       │
       ▼
6. finetune_blip.py   — fine-tune BLIP on the synthetic dataset
       │
       ▼
7. test_inference.py  — evaluate base vs. fine-tuned model via RemoteCLIP cosine similarity
```

---

## Configuration

All paths, API keys, model names, and hyperparameters live in **`config.py`**. Secrets are read from environment variables — never hardcoded.

| Variable | Description |
|---|---|
| `DATA_DIR` | Root directory for all local data files |
| `GOOGLE_MAPS_API_KEY` | Google Maps Static / Street View API key |
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o-mini) |
| `HF_TOKEN` | Hugging Face token for pushing models |
| `S3_BUCKET` | AWS S3 bucket name |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Standard AWS credential env vars |

---

## Setup

```bash
pip install gdal pillow pyproj requests pandas matplotlib \
            transformers torch torchvision datasets \
            openai boto3 open_clip_torch huggingface_hub tqdm

# For RemoteCLIP evaluation
git clone https://github.com/ChenDelong1999/RemoteCLIP/

# Download LandCover.AI dataset
kaggle datasets download -d adrianboguszewski/landcoverai
unzip landcoverai.zip -d data/lcai_tifs
```

Then set your environment variables and run each script in order:

```bash
export GOOGLE_MAPS_API_KEY=...
export OPENAI_API_KEY=...
export HF_TOKEN=...
export S3_BUCKET=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...

python lcai_pipeline.py
python blip_inference.py
python s3_upload.py
python join_tables.py
python text_augmentation.py
python finetune_blip.py
python test_inference.py
```
