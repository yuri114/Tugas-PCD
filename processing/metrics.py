import numpy as np


def calculate_mse(img1, img2):
    """Hitung Mean Squared Error antara dua citra."""
    if img1.shape != img2.shape:
        raise ValueError("Ukuran kedua citra harus sama")

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean((img1 - img2) ** 2)


def calculate_psnr(img1, img2):
    """Hitung Peak Signal-to-Noise Ratio (dB) antara dua citra."""
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)


def calculate_entropy(image):
    """Hitung Shannon entropy (dalam bit) dari citra grayscale."""
    histogram, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]  # hindari log2(0)
    return -np.sum(probabilities * np.log2(probabilities))
