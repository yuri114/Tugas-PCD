import numpy as np

from processing.metrics import calculate_mse, calculate_psnr, calculate_entropy


def test_mse_citra_identik_adalah_nol():
    img = np.full((10, 10), 100, dtype=np.uint8)
    assert calculate_mse(img, img) == 0


def test_mse_citra_berbeda():
    img1 = np.zeros((2, 2), dtype=np.uint8)
    img2 = np.full((2, 2), 10, dtype=np.uint8)
    # selisih tiap pixel = 10, MSE = 10^2 = 100
    assert calculate_mse(img1, img2) == 100


def test_psnr_citra_identik_adalah_inf():
    img = np.full((10, 10), 100, dtype=np.uint8)
    assert calculate_psnr(img, img) == float("inf")


def test_entropy_citra_seragam_adalah_nol():
    # semua pixel sama -> tidak ada variasi informasi -> entropy = 0
    img = np.full((10, 10), 50, dtype=np.uint8)
    assert calculate_entropy(img) == 0


def test_entropy_dalam_rentang_valid():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(50, 50), dtype=np.uint8)
    entropy = calculate_entropy(img)
    assert 0 <= entropy <= 8
