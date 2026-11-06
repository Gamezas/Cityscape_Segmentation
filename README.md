# Segmentación de imágenes de tráfico
Este proyecto es un modelo U-Net para clasificar imágnes de tráfico, el objetivo es análizar la capacidad de U-Net para la predicción
de objetos como autos, señales, bicicletas, edificios, etc.

## Orden de carpetas
El orden para clonar el repositorio es el siguiente

```
Cityscape_Segmentation/
│
├── data/
│   └── Cityscape_Dataset/
│       ├── gtFine/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       │
│       └── leftImg8bit/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/
│   └── unet_cityscapes_final.keras
├── EDA.py
├── test_model.py
├── train.py
├── visualization.py
└── README.md
```

Para los datos es necesario descargarlos de la plataforma Kaggle [Cityscape Dataset: Urban Street Scene Segmentation](https://www.kaggle.com/datasets/electraawais/cityscape-dataset/data)
Se deben colocar las carpetas `gtFine` y `leftImg8bit` como se muestra en la parte de arriba.
