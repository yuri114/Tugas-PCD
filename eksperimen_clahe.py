import matplotlib.pyplot as plt

from utils.image_utils import load_image
from processing.enhancement import apply_he, apply_clahe

img = load_image("assets/contoh_kontras_rendah.jpg")
img_he = apply_he(img)

# Coba ubah dua nilai ini untuk melihat efeknya pada hasil CLAHE (3.3)
clip_limit = 2.0
tile_grid_size = (8, 8)
img_clahe = apply_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

plt.figure(figsize=(12, 8))

# Baris 1: perbandingan gambar
plt.subplot(2, 3, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_he, cmap="gray", vmin=0, vmax=255)
plt.title("HE")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
plt.title(f"CLAHE (clip={clip_limit}, tile={tile_grid_size})")
plt.axis("off")

# Baris 2: perbandingan histogram
plt.subplot(2, 3, 4)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Histogram Original")
plt.xlabel("Nilai Pixel")
plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 5)
plt.hist(img_he.ravel(), bins=256, range=(0, 256))
plt.title("Histogram HE")
plt.xlabel("Nilai Pixel")
plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 6)
plt.hist(img_clahe.ravel(), bins=256, range=(0, 256))
plt.title("Histogram CLAHE")
plt.xlabel("Nilai Pixel")
plt.ylabel("Jumlah Pixel")

plt.tight_layout()
plt.show()
