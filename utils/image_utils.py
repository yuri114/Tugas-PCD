import cv2


def load_image(filepath):
    """Baca gambar dari filepath sebagai grayscale (matriks 2D uint8)."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {filepath}")
    return img
