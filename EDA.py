from pathlib import Path
from PIL import Image
import pandas as pd

# Ruteo base para explorar las carpetas de forma local
DATASET_PATH = Path("data/Cityscape_Dataset")

image_root = DATASET_PATH / "leftImg8bit"
mask_root = DATASET_PATH / "gtFine"

# Exploración de la distribución de la partición 
folders = ["test", "train", "val"]

print("===== Distribución de imágenes =====")

for folder in folders:

    image_dir = Path(image_root / folder)
    mask_dir = Path(mask_root / folder)

    images = list(image_dir.rglob("*_leftImg8bit.png"))
    masks = list(mask_dir.rglob("*_gtFine_labelIds.png"))
    color = list(mask_dir.rglob("*_gtFine_color.png"))

    print(f"\n{folder.upper()}")
    print(f"Images: {len(images)}")
    print(f"Masks labelIds: {len(masks)}")
    print(f"Color: {len(color)}")

# Formato de las imágenes
print("===== Formato de las imágenes =====")

image_paths = list(
    (image_root / "train").rglob("*_leftImg8bit.png")
)

image_info = []

for path in image_paths:

    with Image.open(path) as img:

        image_info.append({
            "filename": path.name,
            "width": img.width,
            "height": img.height,
            "channels": len(img.getbands()),
            "mode": img.mode,
            "format": img.format
        })

df_images = pd.DataFrame(image_info)
print(df_images.head())

print("\nEstadisticas de las Dimensiones:")
print(df_images[["width", "height"]].describe())

print("\nDimensiones únicas:")
print(
    df_images.groupby(["width", "height"]).size()
)

print("===== Formato de las máscaras =====")

mask_paths = list(
    (mask_root / "train").rglob("*_gtFine_labelIds.png")
)

mask_info = []

for path in mask_paths:

    with Image.open(path) as img:

        mask_info.append({
            "filename": path.name,
            "width": img.width,
            "height": img.height,
            "channels": len(img.getbands()),
            "mode": img.mode,
            "format": img.format
        })

df_masks = pd.DataFrame(mask_info)
print(df_masks.head())

print("\nEstadisticas de las Dimensiones:")
print(df_masks[["width", "height"]].describe())

print("\nDimensiones únicas:")
print(
    df_masks.groupby(["width", "height"]).size()
)

# Revisar que cada imágen tenga una máscara
print("===== Par de imágen y máscara =====")

missing_masks = []

for image_path in image_paths:

    mask_name = image_path.name.replace(
        "_leftImg8bit.png",
        "_gtFine_labelIds.png"
    )

    mask_path = (
        mask_root / "train" /
        image_path.parent.relative_to(image_root / "train") /
        mask_name
    )

    if not mask_path.exists():
        missing_masks.append(image_path.name)

print("Total images:", len(image_paths))
print("Missing masks:", len(missing_masks))

if missing_masks:
    print(missing_masks[:10])