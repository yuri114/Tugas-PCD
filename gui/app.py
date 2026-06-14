import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from utils.image_utils import load_image
from processing.enhancement import apply_he, apply_clahe
from processing.metrics import calculate_mse, calculate_psnr, calculate_entropy

UKURAN_PREVIEW = (250, 250)
CLIP_LIMIT_DEFAULT = "2.0"
TILE_SIZE_DEFAULT = "8"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Peningkatan Kualitas Citra - HE vs CLAHE")

        self.img_original = None
        self.img_he = None
        self.img_clahe = None

        self._buat_widget()

    def _buat_widget(self):
        frame_param = tk.Frame(self)
        frame_param.pack(pady=5)

        tk.Label(frame_param, text="Clip Limit:").pack(side=tk.LEFT, padx=5)
        self.entry_clip_limit = tk.Entry(frame_param, width=5)
        self.entry_clip_limit.insert(0, CLIP_LIMIT_DEFAULT)
        self.entry_clip_limit.pack(side=tk.LEFT)

        tk.Label(frame_param, text="Tile Size:").pack(side=tk.LEFT, padx=5)
        self.entry_tile_size = tk.Entry(frame_param, width=5)
        self.entry_tile_size.insert(0, TILE_SIZE_DEFAULT)
        self.entry_tile_size.pack(side=tk.LEFT)

        frame_tombol = tk.Frame(self)
        frame_tombol.pack(pady=10)

        tk.Button(frame_tombol, text="Muat Gambar", command=self.muat_gambar).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(frame_tombol, text="Proses", command=self.proses_gambar).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(frame_tombol, text="Simpan Hasil", command=self.simpan_hasil).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(frame_tombol, text="Reset", command=self.reset).pack(
            side=tk.LEFT, padx=5
        )

        frame_gambar = tk.Frame(self)
        frame_gambar.pack(padx=10, pady=10)

        self.label_original, self.label_metrik_original = self._buat_panel(
            frame_gambar, "Original", 0
        )
        self.label_he, self.label_metrik_he = self._buat_panel(frame_gambar, "HE", 1)
        self.label_clahe, self.label_metrik_clahe = self._buat_panel(
            frame_gambar, "CLAHE", 2
        )

    def _buat_panel(self, parent, judul, kolom):
        frame = tk.Frame(parent)
        frame.grid(row=0, column=kolom, padx=10)

        tk.Label(frame, text=judul, font=("Arial", 10, "bold")).pack()
        label_gambar = tk.Label(frame)
        label_gambar.pack()
        label_metrik = tk.Label(frame, justify=tk.LEFT, anchor="w")
        label_metrik.pack(pady=5)
        return label_gambar, label_metrik

    def muat_gambar(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not filepath:
            return

        self.img_original = load_image(filepath)
        self.img_he = None
        self.img_clahe = None

        self._tampilkan_gambar(self.img_original, self.label_original)
        self.label_metrik_original.configure(
            text=f"Entropy: {calculate_entropy(self.img_original):.4f}"
        )
        self._kosongkan_panel(self.label_he, self.label_metrik_he)
        self._kosongkan_panel(self.label_clahe, self.label_metrik_clahe)

    def _ambil_parameter_clahe(self):
        try:
            clip_limit = float(self.entry_clip_limit.get())
            tile_size = int(self.entry_tile_size.get())
            if clip_limit <= 0 or tile_size <= 0:
                raise ValueError
            return clip_limit, (tile_size, tile_size)
        except ValueError:
            messagebox.showerror(
                "Parameter Tidak Valid",
                "Clip Limit harus angka > 0 dan Tile Size harus bilangan bulat > 0.",
            )
            return None

    def proses_gambar(self):
        if self.img_original is None:
            messagebox.showwarning("Belum Ada Gambar", "Muat gambar terlebih dahulu.")
            return

        parameter = self._ambil_parameter_clahe()
        if parameter is None:
            return
        clip_limit, tile_grid_size = parameter

        self.img_he = apply_he(self.img_original)
        self.img_clahe = apply_clahe(self.img_original, clip_limit, tile_grid_size)

        self._tampilkan_gambar(self.img_he, self.label_he)
        self._tampilkan_gambar(self.img_clahe, self.label_clahe)

        self._tampilkan_metrik(self.img_he, self.label_metrik_he)
        self._tampilkan_metrik(self.img_clahe, self.label_metrik_clahe)

    def _tampilkan_metrik(self, img_hasil, label_target):
        mse = calculate_mse(self.img_original, img_hasil)
        psnr = calculate_psnr(self.img_original, img_hasil)
        entropy = calculate_entropy(img_hasil)
        label_target.configure(
            text=f"MSE: {mse:.2f}\nPSNR: {psnr:.2f} dB\nEntropy: {entropy:.4f}"
        )

    def simpan_hasil(self):
        if self.img_he is None or self.img_clahe is None:
            messagebox.showwarning("Belum Ada Hasil", "Proses gambar terlebih dahulu.")
            return

        folder = filedialog.askdirectory(title="Pilih Folder untuk Menyimpan Hasil")
        if not folder:
            return

        cv2.imwrite(os.path.join(folder, "original.png"), self.img_original)
        cv2.imwrite(os.path.join(folder, "hasil_he.png"), self.img_he)
        cv2.imwrite(os.path.join(folder, "hasil_clahe.png"), self.img_clahe)
        messagebox.showinfo("Berhasil", f"Hasil disimpan di:\n{folder}")

    def reset(self):
        self.img_original = None
        self.img_he = None
        self.img_clahe = None

        self._kosongkan_panel(self.label_original, self.label_metrik_original)
        self._kosongkan_panel(self.label_he, self.label_metrik_he)
        self._kosongkan_panel(self.label_clahe, self.label_metrik_clahe)

        self.entry_clip_limit.delete(0, tk.END)
        self.entry_clip_limit.insert(0, CLIP_LIMIT_DEFAULT)
        self.entry_tile_size.delete(0, tk.END)
        self.entry_tile_size.insert(0, TILE_SIZE_DEFAULT)

    def _kosongkan_panel(self, label_gambar, label_metrik):
        label_gambar.configure(image="")
        label_gambar.image = None
        label_metrik.configure(text="")

    def _tampilkan_gambar(self, img_array, label_target):
        img_pil = Image.fromarray(img_array)
        img_pil.thumbnail(UKURAN_PREVIEW)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_target.configure(image=img_tk)
        label_target.image = img_tk  # simpan referensi, hindari garbage collector


if __name__ == "__main__":
    app = App()
    app.mainloop()
