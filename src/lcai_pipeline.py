"""
Step 1: Split LandCover.AI orthophotos into 512x512 patches, compute geographic coordinates
        for each patch, fetch the nearest Google Street View image, and visualize alignment.

Prerequisites:
    pip install gdal pillow pyproj requests pandas matplotlib
    Download the LandCover.AI dataset: kaggle datasets download -d adrianboguszewski/landcoverai
"""

import glob
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
from pyproj import Proj, Transformer

import config


# ---------------------------------------------------------------------------
# 1. Split TIF patches and record geographic coordinates
# ---------------------------------------------------------------------------

def split_and_store_with_geographic_coords(input_folder, output_folder, txt_output):
    from osgeo import gdal

    os.makedirs(output_folder, exist_ok=True)

    with open(txt_output, "w") as coord_file:
        coord_file.write("image_id,upper_left_lat_dec,upper_left_lon_dec,lower_right_lat_dec,lower_right_lon_dec\n")

        for filename in os.listdir(input_folder):
            if not filename.endswith(".tif"):
                continue

            filepath = os.path.join(input_folder, filename)
            dataset = gdal.Open(filepath)
            geotransform = dataset.GetGeoTransform()
            origin_x, pixel_width, _, origin_y, _, pixel_height = geotransform

            img = Image.open(filepath)
            img_width, img_height = img.size

            in_proj = Proj(init="epsg:2180")
            out_proj = Proj(init="epsg:4326")
            transformer = Transformer.from_proj(in_proj, out_proj, always_xy=True)

            for i in range(0, img_width, 512):
                for j in range(0, img_height, 512):
                    subimage = img.crop((i, j, min(i + 512, img_width), min(j + 512, img_height)))
                    subimage_id = f"{filename[:-4]}_{i}_{j}.tif"
                    subimage.save(os.path.join(output_folder, subimage_id))

                    upper_left_x = origin_x + i * pixel_width
                    upper_left_y = origin_y + j * pixel_height
                    lower_right_x = upper_left_x + min(512, img_width - i) * pixel_width
                    lower_right_y = upper_left_y + min(512, img_height - j) * pixel_height

                    upper_left_lon, upper_left_lat = transformer.transform(upper_left_x, upper_left_y)
                    lower_right_lon, lower_right_lat = transformer.transform(lower_right_x, lower_right_y)

                    coord_file.write(
                        f"{subimage_id},{upper_left_lat},{upper_left_lon},{lower_right_lat},{lower_right_lon}\n"
                    )

            dataset = None


# ---------------------------------------------------------------------------
# 2. Fetch Street View images for each patch
# ---------------------------------------------------------------------------

def process_and_save_street_view_images(txt_input, output_folder, api_key):
    os.makedirs(output_folder, exist_ok=True)
    updated_lines = []

    with open(txt_input, "r") as coord_file:
        header = coord_file.readline().strip()
        updated_lines.append(header + ",distance\n")

        for line in coord_file:
            parts = line.strip().split(",")
            image_id = parts[0]
            upper_left_lat = round(float(parts[1]), 6)
            upper_left_lon = round(float(parts[2]), 6)
            lower_right_lat = round(float(parts[3]), 6)
            lower_right_lon = round(float(parts[4]), 6)

            center_lat, center_lon = _calculate_center(
                (upper_left_lat, upper_left_lon),
                (upper_left_lat, upper_left_lon),
                (lower_right_lat, lower_right_lon),
                (lower_right_lat, lower_right_lon),
            )

            nearest_lat, nearest_lon, distance = _find_nearest_street_view(
                center_lat, center_lon, api_key, radius=config.STREET_VIEW_RADIUS
            )

            if nearest_lat and nearest_lon:
                save_path = os.path.join(output_folder, f"{image_id}.jpg")
                _get_and_save_street_view_image(nearest_lat, nearest_lon, center_lat, center_lon, save_path, api_key)
                updated_lines.append(f"{line.strip()},{distance:.2f}\n")
            else:
                print(f"No Street View found for {image_id}.")
                updated_lines.append(f"{line.strip()},N/A\n")

    with open(txt_input, "w") as coord_file:
        coord_file.writelines(updated_lines)


def _calculate_center(upper_left, lower_left, upper_right, lower_right):
    avg_lat = (upper_left[0] + lower_left[0] + upper_right[0] + lower_right[0]) / 4
    avg_lon = (upper_left[1] + lower_left[1] + upper_right[1] + lower_right[1]) / 4
    return avg_lat, avg_lon


def _find_nearest_street_view(center_lat, center_lon, api_key, radius=2000):
    url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata?"
        f"location={center_lat},{center_lon}&radius={radius}&source=outdoor&key={api_key}"
    )
    resp = requests.get(url).json()
    if resp.get("status") == "OK":
        lat = resp["location"]["lat"]
        lon = resp["location"]["lng"]
        return lat, lon, _haversine(center_lat, center_lon, lat, lon)
    return None, None, None


def _haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _get_and_save_street_view_image(nearest_lat, nearest_lon, ref_lat, ref_lon, save_path, api_key):
    heading = _calculate_bearing(nearest_lat, nearest_lon, ref_lat, ref_lon)
    url = (
        f"https://maps.googleapis.com/maps/api/streetview?"
        f"size=512x512&location={nearest_lat},{nearest_lon}&heading={heading}&key={api_key}"
    )
    with open(save_path, "wb") as f:
        f.write(requests.get(url).content)


# ---------------------------------------------------------------------------
# 3. Visualize alignment quality (top-3 closest pairs)
# ---------------------------------------------------------------------------

def verify_alignment(txt_input, street_view_folder, tiff_folder, n=3):
    entries = []
    with open(txt_input, "r") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            if parts[-1] != "N/A":
                entries.append((parts[0], float(parts[-1])))

    entries.sort(key=lambda x: x[1])
    for image_id, distance in entries[:n]:
        base_id = image_id[:-4]
        tiff_paths = glob.glob(os.path.join(tiff_folder, f"{base_id}.*"))
        sv_paths = glob.glob(os.path.join(street_view_folder, f"{base_id}.*"))
        if not tiff_paths or not sv_paths:
            print(f"Missing files for {base_id}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"{base_id}  —  distance: {distance:.0f} m")
        axes[0].imshow(Image.open(tiff_paths[0]))
        axes[0].set_title("Satellite patch")
        axes[0].axis("off")
        axes[1].imshow(Image.open(sv_paths[0]))
        axes[1].set_title("Street View")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 4. Distance distribution analysis
# ---------------------------------------------------------------------------

def analyze_distances(txt_input):
    df = pd.read_csv(txt_input)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    num_na = df["distance"].isna().sum()
    num_valid = df["distance"].notna().sum()

    plt.figure(figsize=(8, 4))
    plt.bar(["Valid", "N/A"], [num_valid, num_na], color=["green", "red"])
    plt.title("Valid vs N/A Street View distances")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    valid = df["distance"].dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(valid, bins=30, color="steelblue", edgecolor="black")
    plt.title("Distribution of Street View distances")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Splitting TIF patches...")
    split_and_store_with_geographic_coords(
        config.INPUT_TIFF_FOLDER,
        config.TIFF_PATCHES_FOLDER,
        config.COORDS_TXT,
    )

    print("Fetching Street View images...")
    process_and_save_street_view_images(
        config.COORDS_TXT,
        config.STREET_IMAGES_FOLDER,
        config.GOOGLE_MAPS_API_KEY,
    )

    print("Verifying alignment...")
    verify_alignment(config.COORDS_TXT, config.STREET_IMAGES_FOLDER, config.TIFF_PATCHES_FOLDER)

    print("Analyzing distance distribution...")
    analyze_distances(config.COORDS_TXT)
