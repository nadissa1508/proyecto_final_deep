"""Transformaciones de imagen para entrenamiento y evaluación.

Entrenamiento: recorte aleatorio (≥ 60 % del área, para no perder las lesiones pequeñas),
giros horizontal y vertical (la orientación de la hoja es arbitraria), rotación de 15° y
variación leve de brillo/contraste. El tono (hue) casi no se toca: el naranja de las
pústulas es parte de la señal. Evaluación: solo redimensionar y normalizar.
"""

from __future__ import annotations

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
