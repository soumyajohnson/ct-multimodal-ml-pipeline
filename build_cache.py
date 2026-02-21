import os
import numpy as np

# pylidc sometimes expects np.int in older codepaths
if not hasattr(np, "int"):
    np.int = int

import matplotlib.pyplot as plt
import pylidc as pl
from scipy.ndimage import zoom

# =========================================================
# SETTINGS
# =========================================================
CROP_SIZE = 64
TARGET_CROPS = 500
SCAN_LIMIT = 1200
ISO_SPACING_MM = 1.0

# Where to save cache (SSD!)
CACHE_PATH = r"your_cache_path_here"
# Example (Windows): r"E:\lidc_cache\lidc_ct_cache_iso_fp16_500.npz"
# Example (Mac): "/Volumes/SSD/lidc_cache/lidc_ct_cache_iso_fp16_500.npz"

OUTDIR = "outputs_cache"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

print("Scans in DB:", pl.query(pl.Scan).count())

# =========================================================
# HELPERS
# =========================================================
def hu_norm(vol):
    vol = np.clip(vol, -1000, 400)
    vol = (vol + 1000) / 1400.0
    return vol.astype(np.float32)

def get_scan_spacing_mm(scan):
    z = getattr(scan, "slice_spacing", None)
    xy = getattr(scan, "pixel_spacing", None)
    if z is None or xy is None:
        return (1.0, 1.0, 1.0)
    try:
        return (float(z), float(xy[0]), float(xy[1]))
    except Exception:
        return (1.0, 1.0, 1.0)

def resample_isotropic(vol_zyx, spacing_zyx, new_spacing=1.0):
    spacing_zyx = np.array(spacing_zyx, dtype=np.float32)
    new_spacing_zyx = np.array([new_spacing, new_spacing, new_spacing], dtype=np.float32)
    zoom_factors = spacing_zyx / new_spacing_zyx
    vol_rs = zoom(vol_zyx, zoom=zoom_factors, order=1)
    return vol_rs, zoom_factors

def crop_around_center(vol_zyx, center_zyx, size):
    zc, yc, xc = center_zyx
    zc, yc, xc = int(round(zc)), int(round(yc)), int(round(xc))
    half = size // 2
    pad = half + 2

    vol_pad = np.pad(vol_zyx, ((pad, pad), (pad, pad), (pad, pad)), mode="edge")

    zc += pad
    yc += pad
    xc += pad

    z0, z1 = zc - half, zc + half
    y0, y1 = yc - half, yc + half
    x0, x1 = xc - half, xc + half

    crop = vol_pad[z0:z1, y0:y1, x0:x1]

    if crop.shape != (size, size, size):
        crop = crop[:size, :size, :size]
        crop = np.pad(
            crop,
            (
                (0, size - crop.shape[0]),
                (0, size - crop.shape[1]),
                (0, size - crop.shape[2]),
            ),
            mode="edge",
        )
    return crop

# =========================================================
# BUILD CACHE
# =========================================================
X, y, pids = [], [], []
saved_examples = False

all_scans = pl.query(pl.Scan).limit(SCAN_LIMIT).all()
print("Queried scans:", len(all_scans))

for s in all_scans:
    if len(X) >= TARGET_CROPS:
        break

    try:
        vol = s.to_volume()  # (Z,Y,X)
    except Exception as e:
        print("Skipping scan:", getattr(s, "patient_id", "UNKNOWN"), "| reason:", e)
        continue

    spacing_zyx = get_scan_spacing_mm(s)
    vol_rs, zoom_factors = resample_isotropic(vol, spacing_zyx, new_spacing=ISO_SPACING_MM)

    if not saved_examples:
        zmid = vol_rs.shape[0] // 2
        img = hu_norm(vol_rs[zmid])
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"{s.patient_id} | CT mid slice (iso {ISO_SPACING_MM}mm)")
        plt.savefig(os.path.join(OUTDIR, "example_ct_mid_slice.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved:", os.path.join(OUTDIR, "example_ct_mid_slice.png"))

    nodules = s.cluster_annotations()
    if not nodules:
        continue

    for cluster in nodules:
        m = float(np.mean([ann.malignancy for ann in cluster]))

        if m <= 2.0:
            label = 0
        elif m >= 4.0:
            label = 1
        else:
            continue

        # consensus centroid
        try:
            cents = np.array([ann.centroid for ann in cluster], dtype=np.float32)
            cent = cents.mean(axis=0)   # (z,y,x) in original voxels
        except Exception:
            continue

        # map centroid to resampled
        cent_rs = cent * zoom_factors

        crop = crop_around_center(vol_rs, cent_rs, CROP_SIZE)
        crop = hu_norm(crop)

        if not saved_examples:
            mid = CROP_SIZE // 2
            plt.figure()
            plt.imshow(crop[mid], cmap="gray")
            plt.axis("off")
            plt.title(f"{s.patient_id} | nodule crop | mal={m:.2f} | label={label}")
            plt.savefig(os.path.join(OUTDIR, "example_nodule_crop.png"), dpi=200, bbox_inches="tight")
            plt.close()

            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1); plt.imshow(crop[mid, :, :], cmap="gray"); plt.title("Axial"); plt.axis("off")
            plt.subplot(1, 3, 2); plt.imshow(crop[:, mid, :], cmap="gray"); plt.title("Coronal"); plt.axis("off")
            plt.subplot(1, 3, 3); plt.imshow(crop[:, :, mid], cmap="gray"); plt.title("Sagittal"); plt.axis("off")
            plt.suptitle(f"{s.patient_id} | mal={m:.2f} | label={label}")
            plt.savefig(os.path.join(OUTDIR, "example_crop_ortho.png"), dpi=200, bbox_inches="tight")
            plt.close()

            print("Saved examples in:", OUTDIR)
            saved_examples = True

        X.append(crop)
        y.append(label)
        pids.append(s.patient_id)

        if len(X) % 25 == 0:
            print("Crops so far:", len(X))

        if len(X) >= TARGET_CROPS:
            break

print("Extracted crops:", len(X))
if len(X) < 20:
    raise RuntimeError("Too few crops extracted. Increase SCAN_LIMIT or check pylidc DB.")

# store compact on SSD
X = np.stack(X)[:, None, :, :, :].astype(np.float16)  # fp16 cache
y = np.array(y, dtype=np.int64)
pids = np.array(pids)

np.savez_compressed(CACHE_PATH, Xct=X, y=y, groups=pids)

size_mb = os.path.getsize(CACHE_PATH) / 1e6
print(f"✅ Saved cache to: {CACHE_PATH}  ({size_mb:.1f} MB)")
print("Example images saved to:", OUTDIR)
print("Cache arrays:", X.shape, X.dtype, "| y:", y.shape, "| groups:", pids.shape)
