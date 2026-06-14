import cv2
import numpy as np


def apply_he(image):
    """Terapkan Histogram Equalization (HE) pada citra grayscale.

    HE meratakan distribusi intensitas pixel menggunakan CDF (Cumulative
    Distribution Function) sehingga kontras citra meningkat.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Citra harus berupa array grayscale 2D")
    if image.dtype != np.uint8:
        raise ValueError("Citra harus bertipe uint8 (0-255)")

    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Berbeda dari HE biasa, CLAHE membagi citra menjadi tile-tile kecil dan
    menerapkan HE secara lokal pada tiap tile, dengan clip_limit untuk
    membatasi penguatan kontras berlebih (mencegah noise menguat).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("Citra harus berupa array grayscale 2D")
    if image.dtype != np.uint8:
        raise ValueError("Citra harus bertipe uint8 (0-255)")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)
