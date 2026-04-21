"""
Step 2: Join the coordinates table (lcai_coords.txt) with the BLIP-generated
        descriptions (image_descriptions.csv) into a single CSV for augmentation.

Prerequisites:
    pip install pandas
"""

import pandas as pd

import config


def join_tables(coords_path, desc_path, output_path):
    coords_df = pd.read_csv(coords_path, delimiter=",")
    coords_df.columns = [c.strip() for c in coords_df.columns]

    desc_df = pd.read_csv(desc_path)
    desc_df["image_name"] = desc_df["image_name"].str.replace("tif_patches/", "", regex=False)

    joined = coords_df.merge(desc_df, left_on="image_id", right_on="image_name", how="inner")
    print(f"Joined shape: {joined.shape}")
    print(joined.head())

    joined.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    join_tables(config.COORDS_TXT, config.DESCRIPTIONS_CSV, config.JOINED_CSV)
