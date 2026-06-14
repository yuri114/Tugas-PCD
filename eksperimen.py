import cv2
import matplotlib.pyplot as plt

from utils.image_utils import load_image

# Baca gambar grayscale & lihat ukuran (shape) dan tipe datanya (dtype)
img = load_image("assets/contoh_kontras_rendah.jpg")

print("Shape:", img.shape)   # (tinggi, lebar) dalam pixel
print("Dtype:", img.dtype)   # tipe data tiap pixel, harus uint8 (0-255)

# Tampilkan gambar dalam jendela
cv2.imshow("Gambar Original", img)
cv2.waitKey(0)            # tunggu sampai user menekan tombol apapun
cv2.destroyAllWindows()   # tutup semua jendela gambar

# Akses & ubah pixel individual
# img[baris, kolom] -> baris = posisi vertikal (y), kolom = posisi horizontal (x)
print("Pixel di pojok kiri atas (0,0):", img[0, 0])

tengah_y = img.shape[0] // 2
tengah_x = img.shape[1] // 2
print("Pixel di tengah gambar:", img[tengah_y, tengah_x])

# Ubah satu pixel menjadi putih penuh (255)
img_modif = img.copy()
img_modif[0, 0] = 255
print("Pixel (0,0) setelah diubah:", img_modif[0, 0])

# Hitung & tampilkan histogram intensitas
# Histogram = grafik yang menunjukkan seberapa sering setiap nilai
# pixel (0-255) muncul di gambar

plt.figure()
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Histogram Intensitas Gambar")
plt.xlabel("Nilai Pixel (0-255)")
plt.ylabel("Jumlah Pixel")
plt.show()
