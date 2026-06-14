import numpy as np
import pytest

from processing.enhancement import apply_he, apply_clahe


def buat_citra_kontras_rendah():
    """Citra sintetis dengan rentang nilai pixel sempit (90-160)."""
    rng = np.random.default_rng(0)
    return rng.integers(90, 160, size=(50, 50), dtype=np.uint8)


def test_apply_he_preserves_shape_and_dtype():
    img = buat_citra_kontras_rendah()
    hasil = apply_he(img)
    assert hasil.shape == img.shape
    assert hasil.dtype == np.uint8


def test_apply_he_meningkatkan_kontras():
    img = buat_citra_kontras_rendah()
    hasil = apply_he(img)
    assert hasil.std() > img.std()


def test_apply_he_menolak_input_bukan_grayscale():
    img_warna = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_he(img_warna)


def test_apply_clahe_preserves_shape_and_dtype():
    img = buat_citra_kontras_rendah()
    hasil = apply_clahe(img)
    assert hasil.shape == img.shape
    assert hasil.dtype == np.uint8


def test_apply_clahe_mengubah_citra():
    img = buat_citra_kontras_rendah()
    hasil = apply_clahe(img)
    assert not np.array_equal(hasil, img)
