from utils.image_utils import load_image
from processing.enhancement import apply_he, apply_clahe
from processing.metrics import calculate_mse, calculate_psnr, calculate_entropy

img = load_image("assets/contoh_kontras_rendah.jpg")
img_he = apply_he(img)
img_clahe = apply_clahe(img)

print("=== Entropy (bit) ===")
print(f"Original : {calculate_entropy(img):.4f}")
print(f"HE       : {calculate_entropy(img_he):.4f}")
print(f"CLAHE    : {calculate_entropy(img_clahe):.4f}")

print("\n=== MSE & PSNR (dibandingkan dengan Original) ===")
print(f"HE    -> MSE: {calculate_mse(img, img_he):.4f}, "
      f"PSNR: {calculate_psnr(img, img_he):.4f} dB")
print(f"CLAHE -> MSE: {calculate_mse(img, img_clahe):.4f}, "
      f"PSNR: {calculate_psnr(img, img_clahe):.4f} dB")

print("\n=== Sanity check: gambar dibandingkan dengan dirinya sendiri ===")
print(f"MSE  : {calculate_mse(img, img):.4f}")
print(f"PSNR : {calculate_psnr(img, img):.4f}")
