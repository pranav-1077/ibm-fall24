import os

# --- Local paths ---
DATA_DIR = os.environ.get("DATA_DIR", "data")
INPUT_TIFF_FOLDER = os.path.join(DATA_DIR, "lcai_tifs")
TIFF_PATCHES_FOLDER = os.path.join(DATA_DIR, "lcai_patches")
STREET_IMAGES_FOLDER = os.path.join(DATA_DIR, "lcai_street_images")
COORDS_TXT = os.path.join(DATA_DIR, "lcai_coords.txt")
DESCRIPTIONS_CSV = os.path.join(DATA_DIR, "image_descriptions.csv")
JOINED_CSV = os.path.join(DATA_DIR, "rsc_joined.csv")
AUGMENTED_CSV = os.path.join(DATA_DIR, "augmented_descriptions.csv")

# --- API keys (set as environment variables) ---
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- AWS / S3 ---
# Credentials are read automatically from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION env vars.
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_TIF_SUBFOLDER = "tif_patches"
S3_STREET_SUBFOLDER = "street_images"

# --- Models ---
BASE_BLIP_MODEL = "Gurveer05/blip-image-captioning-base-rscid-finetuned"
FINETUNED_BLIP_MODEL = "pruhtopia/blip-rsc-5k-1"
CLIP_MODEL_NAME = "RN50"  # options: RN50, ViT-B-32, ViT-L-14
CLIP_CHECKPOINTS_DIR = "checkpoints"

# --- Dataset ---
HF_DATASET = "pruhtopia/rsc-5k-1"
TEST_SPLIT_SIZE = 100

# --- Training ---
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5

# --- Pipeline ---
STREET_VIEW_RADIUS = 2000   # meters, search radius for nearest Street View location
MAX_DISTANCE_FILTER = 500   # meters, max allowed distance between satellite tile and street view
TEXT_AUGMENTATION_WORKERS = 4
