import tensorflow as tf
import glob
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate)
from tensorflow.keras.models import Model

'''
1. Configuración
Ya que las imágenes son de 2048 x 1024 procesarlas así requiere
mucho tiempo y un equipo más potente, es por eso que se re escalan a
256 x 512

Para el training son 2975 imágenes con su respectiva máscara, se trabajará
en batches de 4 para no sobrecargar el equipo de cómputo. Hay un total de
34 clases según la documentación oficial al igual que el repositorio oficial,
sin embargo como se menciona en la documentación son solo 19 para train.

De momento se está asumiendo que los datos están descargados de forma local
por lo que el ruteo seguirá de la siguiente forma

Data/Cityscape_Dataset/ gtFine o leftImg8bit /train

'''
IMG_HEIGHT = 256
IMG_WIDTH = 512
BATCH_SIZE = 8
NUM_CLASSES = 19
EPOCHS = 25

ROOT = Path("data/Cityscape_Dataset")

IMG_DIR = 'Data/Cityscape_Dataset/leftImg8bit/train'
MASK_DIR = 'Data/Cityscape_Dataset/gtFine/train'

print(f"Looking for images in: {IMG_DIR}")
print(f"Looking for masks in:  {MASK_DIR}")

img_paths = sorted(glob.glob(os.path.join(IMG_DIR, '*/*.png')))
mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, '*/*_labelIds.png')))

print(f"Se encontraron {len(img_paths)} imágenes.")
print(f"Se encontraron {len(mask_paths)} máscaras.")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL = MODEL_DIR / "model.keras"

'''
2. Mapeo de clases
Ya que hay 34 clases y solo se necesitan 19, estas se desfasan de lo que aprende el modelo
por lo que es importante mapearlas de forma correcta.

En la documentación oficial se menciona que las clases ignoradas en entrenamiento deben tener
el valor 255 asociado,es por eso que se hace un mapeo y transformación a tensor para que el
modelo pueda ver correctamente las clases de la máscara verdad durante entrenamiento.
'''
id_mapping = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
    17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13,
    27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

label_lut = np.full(256, 255, dtype=np.uint8)

for original_id, train_id in id_mapping.items():
    label_lut[original_id] = train_id

LABEL_LUT = tf.constant(label_lut, dtype=tf.uint8)

'''
3. Emparejamiento de máscara con imágen

Aunque se reviso en el EDA que las máscaras y las imágenes fueran pares, se hace
una revisión adicional por precaución y confirmación de que los datos esten completos

'''

def get_pairs(split):

    image_dir = ROOT / "leftImg8bit" / split
    mask_dir = ROOT / "gtFine" / split

    image_paths = sorted(
        image_dir.rglob("*_leftImg8bit.png")
    )

    pairs = []

    for image_path in image_paths:

        mask_name = image_path.name.replace(
            "_leftImg8bit.png",
            "_gtFine_labelIds.png"
        )

        relative_folder = image_path.parent.relative_to(image_dir)
        mask_path = mask_dir / relative_folder / mask_name

        if mask_path.exists():
            pairs.append((str(image_path), str(mask_path)))

    print(f"{split}: {len(pairs)} image-mask pairs")

    if not pairs:
        raise ValueError(f"No image-mask pairs found for {split}")

    return pairs


train_pairs = get_pairs("train")
val_pairs = get_pairs("val")

'''
4. Carga y preprocesamiento de datos
Ya que hay confirmación de que el emparejamiento fue correcto, se hace la carga
y preprocesamiento de los datos, como se mencionó en un inicio las imagenes se
deben reescalar a 256 x 512 para no sobrecargar el equipo de cómputo.

Tambien se decodifican las imágenes y se convierten en un tensor, las imagenes
tienen 3 canales asociados a los valores RGB.

Las máscaras tienen 1 solo canal ya que cada pixel tiene asignado una de las 
clases, por lo que no se necesitan los valore RGB
'''

def load_data(img_path, mask_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)

    img = tf.image.resize(
        img, (IMG_HEIGHT, IMG_WIDTH)
    )

    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    mask = tf.image.resize(
        mask,
        (IMG_HEIGHT, IMG_WIDTH),
        method="nearest"
    )

    mask = tf.cast(mask, tf.uint8)

    mask = tf.gather(
        LABEL_LUT,
        tf.cast(mask, tf.int32)
    )

    mask = tf.cast(mask, tf.int32)

    mask = tf.squeeze(mask, axis=-1)

    return img, mask

'''
5. Data Augmentation
Para evitar que el modelo aprenda solo de las imagenes proporcionadas
se hara un data augmentation para generar variaciones de las imágenes de training.
'''


def augment(img, mask):

    flip = tf.random.uniform(()) > 0.5

    mask = tf.expand_dims(mask, axis=-1)

    img, mask = tf.cond(
        flip,
        lambda: (
            tf.image.flip_left_right(img),
            tf.image.flip_left_right(mask)
        ),
        lambda: (img, mask)
    )

    mask = tf.squeeze(mask, axis=-1)

    return img, mask


'''
6. TensorFlow Datasets
'''

def make_dataset(pairs, training=False):

    img_paths = [pair[0] for pair in pairs]
    mask_paths = [pair[1] for pair in pairs]

    ds = tf.data.Dataset.from_tensor_slices(
        (img_paths, mask_paths)
    )

    if training:
        ds = ds.shuffle(
            min(len(pairs), 1000)
        )

    ds = ds.map(
        load_data,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        ds = ds.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = make_dataset(train_pairs, training=True)
val_ds = make_dataset(val_pairs)

print("Pipeline completado")

'''
7. U-Net Model
Se espera una entrada de 256 x 512 x 3.

Después de cada pooling se duplica la cantidad de filtros ya que se
encoge la imágen.

num_clases deberá ser la cantidad definida para train 19.

La salida se esepra que sea 256 x 512 x 19 y se aplica softmax
'''

def build_unet_model(input_shape, num_classes):

    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(16, 3, activation="relu", padding="same")(inputs)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(32, 3, activation="relu", padding="same")(p1)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(64, 3, activation="relu", padding="same")(p2)
    p3 = MaxPooling2D(2)(c3)

    c4 = Conv2D(128, 3, activation="relu", padding="same")(p3)
    p4 = MaxPooling2D(2)(c4)

    # Latente
    c5 = Conv2D(256, 3, activation="relu", padding="same")(p4)

    # Decoder
    u6 = Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, 3, activation="relu", padding="same")(u6)

    u7 = Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, 3, activation="relu", padding="same")(u7)

    u8 = Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, 3, activation="relu", padding="same")(u8)

    u9 = Conv2DTranspose(16, 2, strides=2, padding="same")(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, 3, activation="relu", padding="same")(u9)

    # Salida
    outputs = Conv2D(
        num_classes, 1, activation="softmax"
    )(c9)

    return Model(inputs, outputs, name="U_Net_Simple")

'''
8. Loss function
Todas las calses que se deben ignorar se marcan con 255, por lo que es
importante ignorar dichas clases. Al marcarlas como true/false en valid
se pueden transformar 1 y 0, al multiplicarlo con los valores loss, se
anulan los valores de clases inválidas.
'''
def masked_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int32)

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


'''
9. Compilación del modelo
'''

unet = build_unet_model(
    (IMG_HEIGHT, IMG_WIDTH, 3),
    NUM_CLASSES
)

unet.compile(
    optimizer="adam",
    loss=masked_loss
)

unet.summary()


'''
10. Callbacks
El checkpoint guarda el modelo con Loss más bajo en val
EarlyStopping detiene el modelo si deja de mejorar
'''

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]


'''
11. Train
'''

history = unet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

'''
12. Guardar el modelo
'''

unet.save(MODEL_DIR / "unet_cityscapes_final.keras")

print("Training completado")
print(f"Modelo guardado en: {BEST_MODEL}")
