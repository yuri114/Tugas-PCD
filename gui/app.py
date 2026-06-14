import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from utils.image_utils import load_image
from processing.enhancement import apply_he, apply_clahe

UKURAN_PREVIEW = (250, 250)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Peningkatan Kualitas Citra - HE vs CLAHE")

        self.img_original = None  # citra asli (numpy array grayscale)

        self._buat_widget()

    def _buat_widget(self):
        frame_tombol = tk.Frame(self)
        frame_tombol.pack(pady=10)

        tk.Button(frame_tombol, text="Muat Gambar", command=self.muat_gambar).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(frame_tombol, text="Proses", command=self.proses_gambar).pack(
            side=tk.LEFT, padx=5
        )

        frame_gambar = tk.Frame(self)
        frame_gambar.pack(padx=10, pady=10)

        self.label_original = self._buat_panel(frame_gambar, "Original", 0)
        self.label_he = self._buat_panel(frame_gambar, "HE", 1)
        self.label_clahe = self._buat_panel(frame_gambar, "CLAHE", 2)

    def _buat_panel(self, parent, judul, kolom):
        frame = tk.Frame(parent)
        frame.grid(row=0, column=kolom, padx=10)

        tk.Label(frame, text=judul).pack()
        label_gambar = tk.Label(frame)
        label_gambar.pack()
        return label_gambar

    def muat_gambar(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not filepath:
            return

        self.img_original = load_image(filepath)
        self._tampilkan_gambar(self.img_original, self.label_original)

    def proses_gambar(self):
        if self.img_original is None:
            return

        img_he = apply_he(self.img_original)
        img_clahe = apply_clahe(self.img_original)

        self._tampilkan_gambar(img_he, self.label_he)
        self._tampilkan_gambar(img_clahe, self.label_clahe)

    def _tampilkan_gambar(self, img_array, label_target):
        img_pil = Image.fromarray(img_array)
        img_pil.thumbnail(UKURAN_PREVIEW)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_target.configure(image=img_tk)
        label_target.image = img_tk  # simpan referensi, hindari garbage collector


if __name__ == "__main__":
    app = App()
    app.mainloop()
