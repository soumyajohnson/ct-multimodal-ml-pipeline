import os
import numpy as np
if not hasattr(np, "int"):
    np.int = int

import pylidc as pl

# ----------------------------
# PATHS
# ----------------------------
CT_CACHE_PATH = r"your_ct_cache_path"   # <-- your existing CT cache
OUT_STRUCT_CACHE = r"your_xml_cache_path"  # <-- new cache (SSD)
os.makedirs(os.path.dirname(OUT_STRUCT_CACHE), exist_ok=True)

# ----------------------------
# SETTINGS (MUST match CT cache extraction logic)
# ----------------------------
TARGET_CROPS = 500

# Drop near-malignancy (leaky) semantic attrs:
# Most leaky: subtlety, sphericity, margin, lobulation, spiculation, texture
KEEP_ATTRS = ["internalStructure", "calcification"]  # less leaky

def get_struct_features(cluster, attrs):
    feats = []
    for a in attrs:
        vals = [getattr(ann, a) for ann in cluster if getattr(ann, a) is not None]
        mean_val = float(np.mean(vals)) if vals else 3.0
        feats.append((mean_val - 1.0) / 4.0)  # 1..5 -> 0..1
    return np.array(feats, dtype=np.float32)

# ----------------------------
# LOAD CT CACHE (for sanity)
# ----------------------------
d = np.load(CT_CACHE_PATH, allow_pickle=True)
y_ct = d["y"].astype(np.int64)
groups_ct = d["groups"]
print("CT cache:", d["Xct"].shape, d["Xct"].dtype)
print("y counts:", np.bincount(y_ct))

# ----------------------------
# REBUILD STRUCT FEATURES IN SAME ORDER
# ----------------------------
Xs = []
y_rebuilt = []
groups_rebuilt = []

# IMPORTANT: do NOT shuffle scans; iterate like before
scans = pl.query(pl.Scan).all()  # same as typical extraction
print("Scans in DB:", len(scans))
print("Using attrs:", KEEP_ATTRS)

for s in scans:
    if len(y_rebuilt) >= TARGET_CROPS:
        break

    nodules = s.cluster_annotations()
    if not nodules:
        continue

    for cluster in nodules:
        if len(y_rebuilt) >= TARGET_CROPS:
            break

        m = float(np.mean([ann.malignancy for ann in cluster if ann.malignancy is not None]))
        if m <= 2.0:
            label = 0
        elif m >= 4.0:
            label = 1
        else:
            continue

        Xs.append(get_struct_features(cluster, KEEP_ATTRS))
        y_rebuilt.append(label)
        groups_rebuilt.append(s.patient_id)

    if len(y_rebuilt) % 50 == 0 and len(y_rebuilt) > 0:
        print("Built struct samples:", len(y_rebuilt))

Xs = np.stack(Xs).astype(np.float32)
y_rebuilt = np.array(y_rebuilt, dtype=np.int64)
groups_rebuilt = np.array(groups_rebuilt)

print("Rebuilt struct samples:", len(y_rebuilt))
print("Rebuilt y counts:", np.bincount(y_rebuilt))

# ----------------------------
# ALIGNMENT CHECK (VERY IMPORTANT)
# ----------------------------
same_labels = np.array_equal(y_ct, y_rebuilt)
same_groups = np.array_equal(groups_ct, groups_rebuilt)

print("Label alignment with CT cache:", same_labels)
print("Group alignment with CT cache:", same_groups)

if not same_labels or not same_groups:
    raise RuntimeError(
        "Alignment failed. Your CT cache extraction order differs from this rebuild loop. "
        "We need to regenerate a multimodal cache in one pass (CT + Xs)."
    )

# ----------------------------
# SAVE STRUCT CACHE
# ----------------------------
np.savez_compressed(
    OUT_STRUCT_CACHE,
    Xs=Xs,
    y=y_rebuilt,
    groups=groups_rebuilt,
    attrs=np.array(KEEP_ATTRS, dtype=object)
)

print("✅ Saved aligned struct cache:", OUT_STRUCT_CACHE)
print("Xs shape:", Xs.shape)
