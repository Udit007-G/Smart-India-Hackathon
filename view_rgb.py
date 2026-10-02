from pathlib import Path

import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILE LOCATIONS
# ============================================================

folder = Path(r"C:\Users\HP\Desktop\Assignments\SIH\data")

aoi_file = Path("bharweli_aoi.geojson")


b02_file = next(folder.glob("*_B02_10m.jp2"))
b03_file = next(folder.glob("*_B03_10m.jp2"))
b04_file = next(folder.glob("*_B04_10m.jp2"))


# ============================================================
# LOAD AOI
# ============================================================

aoi = gpd.read_file(aoi_file)

print("AOI CRS:", aoi.crs)

# Sentinel-2 files normally use WGS84 / UTM.
# Reproject AOI to the CRS of the satellite image.
with rasterio.open(b04_file) as src:
    satellite_crs = src.crs

aoi = aoi.to_crs(satellite_crs)

shapes = [feature["geometry"] for feature in aoi.__geo_interface__["features"]]


# ============================================================
# READ ONLY THE AOI FROM EACH BAND
# ============================================================

def read_aoi(filename):
    with rasterio.open(filename) as src:
        image, transform = mask(
            src,
            shapes,
            crop=True
        )

        profile = src.profile.copy()

    return image[0].astype(np.float32), transform, profile


blue, transform, profile = read_aoi(b02_file)
green, _, _ = read_aoi(b03_file)
red, _, _ = read_aoi(b04_file)


print("Cropped image size:", red.shape)


# ============================================================
# IMAGE STRETCHING
# ============================================================

def stretch(img):

    low, high = np.percentile(
        img,
        (2, 98)
    )

    if high <= low:
        return np.zeros_like(img)

    result = (img - low) / (high - low)

    return np.clip(result, 0, 1)


rgb = np.dstack([
    stretch(red),
    stretch(green),
    stretch(blue)
])


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(10, 10))

plt.imshow(rgb)

plt.title(
    "Bharweli, Balaghat - Sentinel-2 RGB"
)

plt.axis("off")

plt.show()