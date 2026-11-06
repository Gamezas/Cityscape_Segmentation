
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image


'''
1. Configuración
'''

ROOT = Path("data/Cityscape_Dataset")

MODEL_PATH = Path("models/unet_cityscapes_final.keras")
IMAGE_SIZE = (256, 512)
NUM_CLASSES = 19
NUM_IMAGES = 5
SPLIT = "val"


'''
2. ID Mapping
'''

id_mapping = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18
}

LABEL_LUT = np.full(256, 255, dtype=np.uint8)

for original_id, train_id in id_mapping.items():
    LABEL_LUT[original_id] = train_id


'''
3. Nombres de clases
'''

CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]


'''
4. Carga del modelo
'''

def masked_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)

    y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank == 4 else y_true

    valid = tf.not_equal(y_true, 255)

    safe_labels = tf.where(
        valid,
        y_true,
        tf.zeros_like(y_true)
    )

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        safe_labels,
        y_pred
    )

    valid = tf.cast(valid, tf.float32)

    loss = loss * valid

    return tf.reduce_sum(loss) / (
        tf.reduce_sum(valid) + 1e-7
    )


print("...Cargando Modelo...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"masked_loss": masked_loss}
)

model.summary()


'''
5. Pares de imágenes y máscaras
'''

image_dir = ROOT / "leftImg8bit" / SPLIT
mask_dir = ROOT / "gtFine" / SPLIT

image_paths = sorted(
    image_dir.rglob("*_leftImg8bit.png")
)

pairs = []

for image_path in image_paths:

    city = image_path.parent.name

    mask_name = image_path.name.replace(
        "_leftImg8bit.png",
        "_gtFine_labelIds.png"
    )

    mask_path = mask_dir / city / mask_name

    if mask_path.exists():
        pairs.append((image_path, mask_path))


print(f"\nSe encontraron {len(pairs)} pares de imágen con máscara")


if not pairs:
    raise RuntimeError("No se encontraron pares.")


'''
6. Preprocesamiento de máscaras e imágenes
'''

def load_image(image_path):

    image = tf.io.read_file(str(image_path))

    image = tf.image.decode_png(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image


def load_mask(mask_path):

    mask = tf.io.read_file(str(mask_path))

    mask = tf.image.decode_png(
        mask,
        channels=1
    )

    mask = tf.image.resize(
        mask,
        IMAGE_SIZE,
        method="nearest"
    )

    mask = tf.cast(mask, tf.uint8)

    mask = tf.squeeze(mask, axis=-1)

    mask = LABEL_LUT[mask.numpy()]

    return mask


'''
7. IoU
'''
def calculate_iou(y_true, y_pred):

    ious = []

    for class_id in range(NUM_CLASSES):

        true_class = (y_true == class_id)
        pred_class = (y_pred == class_id)

        intersection = np.logical_and(
            true_class,
            pred_class
        ).sum()

        union = np.logical_or(
            true_class,
            pred_class
        ).sum()

        if union > 0:
            iou = intersection / union
            ious.append(iou)

    return np.mean(ious) if ious else np.nan

'''
8. predicción y visualización
'''

selected_pairs = pairs[:NUM_IMAGES]

all_ious = []

for index, (image_path, mask_path) in enumerate(selected_pairs):

    print(f"\nTesting image {index + 1}/{len(selected_pairs)}")

    image = load_image(image_path)

    true_mask = load_mask(mask_path)

    image_batch = tf.expand_dims(image, axis=0)

    prediction = model.predict(
        image_batch,
        verbose=0
    )

    predicted_mask = np.argmax(
        prediction[0],
        axis=-1
    ).astype(np.uint8)

    valid = true_mask != 255

    image_iou = calculate_iou(
        true_mask[valid],
        predicted_mask[valid]
    )

    all_ious.append(image_iou)

    print(f"IoU: {image_iou:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image.numpy())
    axes[0].set_title("Original image")

    axes[1].imshow(true_mask, vmin=0, vmax=18, cmap="tab20")
    axes[1].set_title("Ground truth")

    axes[2].imshow(predicted_mask, vmin=0, vmax=18, cmap="tab20")
    axes[2].set_title(f"Prediction (IoU: {image_iou:.4f})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


'''
9. Resultados
'''

print("\n" + "=" * 40)
print("FINAL EVALUATION")
print("=" * 40)

print(f"Images evaluated: {len(all_ious)}")
print(f"Mean IoU: {np.nanmean(all_ious):.4f}")
