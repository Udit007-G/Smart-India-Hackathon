from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

folder = Path(r"C:\Users\HP\Desktop\Assignments\SIH\data")

aoi_file = Path("bharweli_aoi.geojson")

output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)


# ============================================================
# FIND FILES
# ============================================================

b04_file = next(folder.glob("*_B04_10m.jp2"))
b08_file = next(folder.glob("*_B08_10m.jp2"))
b11_file = next(folder.glob("*_B11_20m.jp2"))
scl_file = next(folder.glob("*_SCL_20m.jp2"))


print("B04:", b04_file)
print("B08:", b08_file)
print("B11:", b11_file)
print("SCL:", scl_file)


# ============================================================
# LOAD AOI
# ============================================================

aoi = gpd.read_file(aoi_file)


# ============================================================
# READ B04 / B08 AT 10m
# ============================================================

def read_band(filename):

    with rasterio.open(filename) as src:

        aoi_reprojected = aoi.to_crs(src.crs)

        shapes = [
            feature["geometry"]
            for feature in aoi_reprojected.__geo_interface__["features"]
        ]

        image, transform = mask(
            src,
            shapes,
            crop=True
        )

        profile = src.profile.copy()

        return (
            image[0].astype(np.float32),
            transform,
            profile,
            src.crs
        )


red, transform, profile, crs = read_band(b04_file)

nir, _, _, _ = read_band(b08_file)


# ============================================================
# READ B11
# ============================================================

with rasterio.open(b11_file) as src:

    aoi_b11 = aoi.to_crs(src.crs)

    shapes = [
        feature["geometry"]
        for feature in aoi_b11.__geo_interface__["features"]
    ]

    swir20, transform20 = mask(
        src,
        shapes,
        crop=True
    )

    swir20 = swir20[0].astype(np.float32)

    profile20 = src.profile.copy()


# ============================================================
# RESAMPLE B11 → 10m
# ============================================================

swir10 = np.empty_like(red, dtype=np.float32)

reproject(
    source=swir20,
    destination=swir10,

    src_transform=transform20,
    src_crs=profile20["crs"],

    dst_transform=transform,
    dst_crs=crs,

    resampling=Resampling.bilinear
)


# ============================================================
# READ SCL
# ============================================================

with rasterio.open(scl_file) as src:

    aoi_scl = aoi.to_crs(src.crs)

    shapes = [
        feature["geometry"]
        for feature in aoi_scl.__geo_interface__["features"]
    ]

    scl20, scl_transform = mask(
        src,
        shapes,
        crop=True
    )

    scl20 = scl20[0]


# Resample SCL to 10m using nearest-neighbour
scl10 = np.empty_like(red, dtype=np.uint8)

reproject(
    source=scl20,
    destination=scl10,

    src_transform=scl_transform,
    src_crs=src.crs,

    dst_transform=transform,
    dst_crs=crs,

    resampling=Resampling.nearest
)


# ============================================================
# CLOUD MASK
# ============================================================

# Sentinel-2 SCL classes:
#
# 3  = Cloud shadow
# 8  = Cloud medium probability
# 9  = Cloud high probability
# 10 = Cirrus
# 11 = Snow / ice

bad_pixels = np.isin(
    scl10,
    [3, 8, 9, 10, 11]
)


# ============================================================
# CALCULATE NDVI
# ============================================================

denominator_ndvi = nir + red

ndvi = np.where(
    denominator_ndvi == 0,
    np.nan,
    (nir - red) / denominator_ndvi
)

ndvi[bad_pixels] = np.nan


# ============================================================
# CALCULATE NDMI
# ============================================================

denominator_ndmi = nir + swir10

ndmi = np.where(
    denominator_ndmi == 0,
    np.nan,
    (nir - swir10) / denominator_ndmi
)

ndmi[bad_pixels] = np.nan


# ============================================================
# SAVE FUNCTION
# ============================================================
profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    height=ndvi.shape[0],
    width=ndvi.shape[1],
    transform=transform,
    crs=crs,
    nodata=np.nan
)


def save_raster(filename, data):

    with rasterio.open(
        output_folder / filename,
        "w",
        **profile
    ) as dst:

        dst.write(
            data.astype(np.float32),
            1
        )


save_raster("bharweli_NDVI.tif", ndvi)

save_raster("bharweli_NDMI.tif", ndmi)


print()
print("===================================")
print("Processing complete!")
print("===================================")

print("NDVI saved to:")
print(output_folder / "bharweli_NDVI.tif")

print("NDMI saved to:")
print(output_folder / "bharweli_NDMI.tif")


# ============================================================
# DISPLAY NDVI + NDMI
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)


# NDVI
im1 = axes[0].imshow(
    ndvi,
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)

axes[0].set_title(
    "Bharweli NDVI"
)

axes[0].axis("off")

plt.colorbar(
    im1,
    ax=axes[0],
    fraction=0.046
)


# NDMI
im2 = axes[1].imshow(
    ndmi,
    cmap="BrBG",
    vmin=-1,
    vmax=1
)

axes[1].set_title(
    "Bharweli NDMI"
)

axes[1].axis("off")

plt.colorbar(
    im2,
    ax=axes[1],
    fraction=0.046
)


plt.tight_layout()

plt.show()