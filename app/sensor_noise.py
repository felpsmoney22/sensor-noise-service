# app/sensor_noise.py
import io
import numpy as np
from PIL import Image
import cv2

# Converters sRGB <-> linear approximation
def srgb_to_linear(x):
    a = 0.055
    mask = x <= 0.04045
    linear = np.empty_like(x)
    linear[mask] = x[mask] / 12.92
    linear[~mask] = ((x[~mask] + a) / (1.0 + a)) ** 2.4
    return linear


def linear_to_srgb(x):
    a = 0.055
    mask = x <= 0.0031308
    sr = np.empty_like(x)
    sr[mask] = x[mask] * 12.92
    sr[~mask] = (1.0 + a) * (x[~mask] ** (1.0 / 2.4)) - a
    return sr


def generate_prnu_map(h, w, strength=0.002, seed=None):
    rng = np.random.default_rng(seed)
    prnu = rng.normal(loc=0.0, scale=1.0, size=(h, w))
    prnu = cv2.GaussianBlur(prnu, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0)
    prnu = prnu / (np.std(prnu) + 1e-12)
    return 1.0 + prnu * strength


def add_sensor_noise_pil(pil_img, iso=400, prnu_strength=0.002, read_noise_sigma=2.0, seed=None):
    """
    Recebe PIL.Image (RGB), retorna bytes JPEG.
    iso: controla intensidade do Poisson (maior iso -> mais ruído)
    prnu_strength: intensidade do PRNU multiplicativo
    read_noise_sigma: sigma do ruído Gaussiano (0..255 scale)
    """
    img = pil_img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    lin = srgb_to_linear(arr)  # (h, w, 3)
    h, w, c = lin.shape

    # PRNU multiplicativo
    prnu = generate_prnu_map(h, w, strength=prnu_strength, seed=seed)  # (h, w)
    lin_prnu = lin * prnu[..., None]  # (h, w, 3)

    # Poisson (shot noise) - escala relacionada ao ISO
    scale = max(1.0, 1024.0 / max(1, iso))
    lam = np.clip(lin_prnu * 255.0 * scale, 0, None)  # (h, w, 3), não-negativo

    rng = np.random.default_rng(seed)
    noisy_poisson = rng.poisson(lam).astype(np.float32) / (255.0 * scale)  # (h, w, 3)

    # Read noise (gaussiano)
    read_sigma_norm = read_noise_sigma / 255.0
    gaussian = rng.normal(loc=0.0, scale=read_sigma_norm, size=(h, w, c)).astype(np.float32)

    # ⚠️ Ajuste: sem criar eixo extra
    noisy_lin = noisy_poisson + gaussian
    noisy_lin = np.clip(noisy_lin, 0.0, 1.0)

    out = linear_to_srgb(noisy_lin)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    out_pil = Image.fromarray(out)
    buf = io.BytesIO()
    out_pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf
