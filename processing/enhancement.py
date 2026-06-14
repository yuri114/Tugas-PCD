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
