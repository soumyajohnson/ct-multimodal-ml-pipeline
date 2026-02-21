import os
import numpy as np

# pylidc sometimes expects np.int in older codepaths
if not hasattr(np, "int"):
    np.int = int

import matplotlib.pyplot as plt

import pylidc as pl
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# =========================================================
# 1) PATHS
# =========================================================
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# =========================================================
# 2) SETTINGS
# =========================================================
CROP_SIZE = 64
TARGET_CROPS = 400       # try 200 for quick; 2000+ for better
SCAN_LIMIT = 1200        # how many scans to consider
EPOCHS = 15
BATCH_SIZE = 2
LR = 1e-3
DEVICE = "cpu"           # "cuda" if available

# =========================================================
# 3) OPTIONAL: show pylidc config sanity
# =========================================================
print("Scans in DB:", pl.query(pl.Scan).count())

# =========================================================
# 4) HELPERS
# =========================================================
def center_crop_or_pad(vol, target=CROP_SIZE):
    # vol: (Z,Y,X)
    z, y, x = vol.shape
    pad_z = max(0, target - z)
    pad_y = max(0, target - y)
    pad_x = max(0, target - x)

    if pad_z or pad_y or pad_x:
        vol = np.pad(
            vol,
            ((pad_z // 2, pad_z - pad_z // 2),
             (pad_y // 2, pad_y - pad_y // 2),
             (pad_x // 2, pad_x - pad_x // 2)),
            mode="edge"
        )

    z, y, x = vol.shape
    cz, cy, cx = z // 2, y // 2, x // 2
    return vol[
        cz - target // 2: cz + target // 2,
        cy - target // 2: cy + target // 2,
        cx - target // 2: cx + target // 2
    ]

def hu_norm(crop):
    crop = np.clip(crop, -1000, 400)
    crop = (crop + 1000) / 1400.0
    return crop.astype(np.float32)

# =========================================================
# 5) EXTRACT NODULE-CENTERED CROPS + LABELS (from XML via pylidc)
# =========================================================
X, y, pids = [], [], []
saved_examples = False

all_scans = pl.query(pl.Scan).limit(SCAN_LIMIT).all()
print("Queried scans:", len(all_scans))

for s in all_scans:
    if len(X) >= TARGET_CROPS:
        break

    try:
        vol = s.to_volume()  # from DICOM
    except Exception as e:
        print("Skipping scan:", getattr(s, "patient_id", "UNKNOWN"), "| reason:", e)
        continue

    # save one CT mid-slice
    if not saved_examples:
        zmid = vol.shape[0] // 2
        img = hu_norm(vol[zmid])
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"{s.patient_id} | CT mid slice")
        plt.savefig(os.path.join(OUTDIR, "example_ct_mid_slice.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print("✅ Saved:", os.path.join(OUTDIR, "example_ct_mid_slice.png"))

    # annotations clusters (requires XML-derived DB)
    nodules = s.cluster_annotations()
    if not nodules:
        continue

    for cluster in nodules:
        # malignancy from radiologists (XML)
        m = float(np.mean([ann.malignancy for ann in cluster]))

        # label rule
        if m <= 2.0:
            label = 0
        elif m >= 4.0:
            label = 1
        else:
            continue  # skip ambiguous 3-ish

        # centroid from one annotation in the cluster
        a = cluster[0]
        try:
            zc, yc, xc = map(int, np.round(a.centroid))  # (z,y,x)
        except Exception:
            continue

        half = CROP_SIZE // 2
        z0, z1 = zc - half, zc + half
        y0, y1 = yc - half, yc + half
        x0, x1 = xc - half, xc + half

        z0p, y0p, x0p = max(0, z0), max(0, y0), max(0, x0)
        z1p, y1p, x1p = min(vol.shape[0], z1), min(vol.shape[1], y1), min(vol.shape[2], x1)

        crop = vol[z0p:z1p, y0p:y1p, x0p:x1p]
        if crop.size == 0:
            continue

        crop = center_crop_or_pad(crop, CROP_SIZE)
        crop = hu_norm(crop)

        # save one nodule crop + orthogonal views once
        if not saved_examples:
            mid = CROP_SIZE // 2

            plt.figure()
            plt.imshow(crop[mid], cmap="gray")
            plt.axis("off")
            plt.title(f"{s.patient_id} | nodule | mal={m:.2f} | label={label}")
            plt.savefig(os.path.join(OUTDIR, "example_nodule_crop.png"), dpi=200, bbox_inches="tight")
            plt.close()
            print("✅ Saved:", os.path.join(OUTDIR, "example_nodule_crop.png"))

            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1); plt.imshow(crop[mid, :, :], cmap="gray"); plt.title("Axial")
            plt.subplot(1, 3, 2); plt.imshow(crop[:, mid, :], cmap="gray"); plt.title("Coronal")
            plt.subplot(1, 3, 3); plt.imshow(crop[:, :, mid], cmap="gray"); plt.title("Sagittal")
            plt.suptitle(f"{s.patient_id} | mal={m:.2f} | label={label}")
            plt.savefig(os.path.join(OUTDIR, "example_crop_ortho.png"), dpi=200, bbox_inches="tight")
            plt.close()
            print("✅ Saved:", os.path.join(OUTDIR, "example_crop_ortho.png"))

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
    raise RuntimeError("Too few crops extracted. Increase SCAN_LIMIT/TARGET_CROPS or check pylidc DB.")

os.makedirs("outputs", exist_ok=True)
np.savez(
    "outputs/lidc_ct_cache.npz",
    Xct=X,          # (N,64,64,64)
    y=y,            # labels
    groups=pids     # patient IDs
)

print("✅ Saved CT cache: outputs/lidc_ct_cache.npz")

# =========================================================
# 6) PREP DATA (patient-wise split)
# =========================================================
X = np.stack(X)[:, None, :, :, :]  # (N,1,64,64,64)
y = np.array(y, dtype=np.int64)
pids = np.array(pids)

print("Overall class counts [0,1]:", np.bincount(y))

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=pids))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("Train class counts [0,1]:", np.bincount(y_train))
print("Val class counts   [0,1]:", np.bincount(y_val))

# =========================================================
# 7) Dataset / Dataloader
# =========================================================
class CropDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return self.X[i], self.y[i]

# Balanced sampler (helps a lot with imbalance)
class_counts = np.bincount(y_train)
if len(class_counts) < 2:
    raise RuntimeError("Only one class present in training set. Increase data / relax thresholds.")
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(CropDS(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(CropDS(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# 8) Tiny 3D CNN (unimodal CT branch baseline)
# =========================================================
class Tiny3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

model = Tiny3DCNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

# class-weighted loss (also helps)
cw = class_weights / class_weights.sum() * 2.0
counts = np.bincount(y_train)
print("Train class counts:", counts)

weights = 1.0 / counts
weights = weights / weights.sum() * 2.0
print("Class weights:", weights)

loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
)
print("Class weights used in loss:", cw.tolist())

# =========================================================
# 9) Metrics
# =========================================================
def evaluate(loader):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            p1 = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
            probs.extend(p1.tolist())
            preds.extend(pred.tolist())
            ys.extend(yb.cpu().numpy().tolist())

    ys = np.array(ys)
    preds = np.array(preds)
    acc = (preds == ys).mean()
    f1 = f1_score(ys, preds, zero_division=0)
    try:
        auc = roc_auc_score(ys, np.array(probs))
    except Exception:
        auc = float("nan")
    return acc, f1, auc

# =========================================================
# 10) Train
# =========================================================
print("\nStarting training...")
best_auc = -1.0

for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item() * yb.size(0)

    train_acc, train_f1, train_auc = evaluate(train_loader)
    val_acc, val_f1, val_auc = evaluate(val_loader)

    print(
        f"Epoch {ep:02d}/{EPOCHS} | "
        f"loss={running/len(train_loader.dataset):.4f} | "
        f"train acc={train_acc:.3f} f1={train_f1:.3f} auc={train_auc:.3f} | "
        f"val acc={val_acc:.3f} f1={val_f1:.3f} auc={val_auc:.3f}"
    )

    # save best by val_auc
    if not np.isnan(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), os.path.join(OUTDIR, "best_tiny3dcnn.pt"))

print("\nDone.")
print("Best val AUC:", best_auc)
print("Saved best model to:", os.path.join(OUTDIR, "best_tiny3dcnn.pt"))
print("Saved examples in outputs/: example_ct_mid_slice.png, example_nodule_crop.png, example_crop_ortho.png")
