from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATASET_PATH = Path("data/Cityscape_Dataset")

image_dir = DATASET_PATH / "leftImg8bit" / "train" / "aachen"
mask_dir = DATASET_PATH / "gtFine" / "train" / "aachen"

images = sorted(image_dir.glob("*_leftImg8bit.png"))

fig, axes = plt.subplots(4, 2, figsize=(16, 20))

for i, image_path in enumerate(images[:4]):

    mask_name = image_path.name.replace(
        "_leftImg8bit.png",
        "_gtFine_color.png"
    )

    mask_path = mask_dir / mask_name

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    axes[i, 0].imshow(image)
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask)
    axes[i, 1].set_title("Mask")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()