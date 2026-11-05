import numpy as np
from src.mri import fft2c, ifft2c, cartesian_mask, simulate_acquisition
from src.classical import zero_filled, tv_pocs
from src.metrics import compute_metrics


def test_fft_roundtrip():
    rng = np.random.default_rng(42)
    image = rng.random((128, 128)).astype(np.float32)
    np.testing.assert_allclose(ifft2c(fft2c(image)).real, image, atol=1e-6)


def test_mask_shape_fraction_center():
    mask = cartesian_mask((128, 128), acceleration=4, seed=2026)
    assert mask.shape == (128, 128)
    assert int(mask[0].sum()) == 32
    assert mask[:, 64].all()
    assert (mask == mask[0]).all()


def test_data_consistency_and_shapes():
    rng = np.random.default_rng(123)
    image = rng.random((128, 128))
    mask = cartesian_mask(image.shape)
    measured, degraded = simulate_acquisition(image, mask)
    assert degraded.shape == image.shape
    np.testing.assert_allclose(measured[mask == 0], 0, atol=1e-10)
    assert zero_filled(measured).shape == image.shape
    result = tv_pocs(measured, mask, steps=2)
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert set(compute_metrics(image, result)) == {'psnr', 'ssim', 'mae'}
