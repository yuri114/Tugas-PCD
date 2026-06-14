import matplotlib.pyplot as plt

from utils.image_utils import load_image
from processing.enhancement import apply_he

img = load_image("assets/contoh_kontras_rendah.jpg")
img_he = apply_he(img)

plt.figure(figsize=(10, 8))

# Gambar original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

# Gambar setelah HE
plt.subplot(2, 2, 2)
plt.imshow(img_he, cmap="gray", vmin=0, vmax=255)
plt.title("Hasil Histogram Equalization")
plt.axis("off")

# Histogram sebelum HE
plt.subplot(2, 2, 3)
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Histogram Original")
plt.xlabel("Nilai Pixel")
plt.ylabel("Jumlah Pixel")

# Histogram setelah HE
plt.subplot(2, 2, 4)
plt.hist(img_he.ravel(), bins=256, range=(0, 256))
plt.title("Histogram Setelah HE")
plt.xlabel("Nilai Pixel")
plt.ylabel("Jumlah Pixel")

plt.tight_layout()
plt.show()
