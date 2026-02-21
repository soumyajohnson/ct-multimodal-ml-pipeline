import os, numpy as np
if not hasattr(np, "int"): np.int = int

import pylidc as pl
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# ============================
# SETTINGS
# ============================
CROP_SIZE = 64
TARGET_CROPS = 800
EPOCHS_STRUCT = 25
BATCH_SIZE_CT = 2
BATCH_SIZE_STRUCT = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

CT_CKPT = os.path.join(OUTDIR, "best_tiny3dcnn.pt")
assert os.path.exists(CT_CKPT), "CT model checkpoint missing"

# ============================
# HELPERS
# ============================
def center_crop_or_pad(vol, target=64):
    z,y,x = vol.shape
    pad = [(0, max(0, target - z)),
           (0, max(0, target - y)),
           (0, max(0, target - x))]
    vol = np.pad(vol,
        [(p[0]//2, p[1]-p[0]//2) for p in pad],
        mode="edge")
    z,y,x = vol.shape
    cz,cy,cx = z//2,y//2,x//2
    return vol[cz-target//2:cz+target//2,
               cy-target//2:cy+target//2,
               cx-target//2:cx+target//2]

def hu_norm(x):
    x = np.clip(x, -1000, 400)
    return ((x + 1000) / 1400).astype(np.float32)

def get_struct_features(cluster):
    attrs = ["subtlety","internalStructure","calcification","sphericity",
             "margin","lobulation","spiculation","texture"]
    feats = []
    for a in attrs:
        vals = [getattr(ann, a) for ann in cluster if getattr(ann, a) is not None]
        feats.append(np.mean(vals) if vals else 3.0)
    feats = np.array(feats, dtype=np.float32)
    return (feats - 1.0) / 4.0  # scale 1–5 → 0–1

def softmax_p1(logits):
    return torch.softmax(logits, dim=1)[:,1]

# ============================
# MODELS
# ============================
class Tiny3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Linear(64,2)

    def forward(self,x):
        return self.classifier(self.features(x).flatten(1))

class StructMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8,32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x): return self.net(x)

# ============================
# LOAD FROZEN CT MODEL
# ============================
ct_model = Tiny3DCNN().to(DEVICE)
ct_model.load_state_dict(torch.load(CT_CKPT, map_location=DEVICE))
ct_model.eval()
for p in ct_model.parameters():
    p.requires_grad = False

s_model = StructMLP().to(DEVICE)

# ============================
# EXTRACT DATA (SLOW BUT CLEAN)
# ============================
Xct, Xs, y, groups = [], [], [], []

scans = pl.query(pl.Scan).all()
print("Scans in DB:", len(scans))

for s in scans:
    if len(y) >= TARGET_CROPS: break
    try:
        vol = s.to_volume()
    except:
        continue

    for cluster in s.cluster_annotations():
        if len(y) >= TARGET_CROPS: break

        m = np.mean([ann.malignancy for ann in cluster if ann.malignancy is not None])
        if m <= 2: label = 0
        elif m >= 4: label = 1
        else: continue

        try:
            zc,yc,xc = map(int, np.round(cluster[0].centroid))
        except:
            continue

        crop = vol[max(0,zc-32):zc+32,
                   max(0,yc-32):yc+32,
                   max(0,xc-32):xc+32]
        if crop.size == 0: continue

        crop = hu_norm(center_crop_or_pad(crop))
        Xct.append(crop)
        Xs.append(get_struct_features(cluster))
        y.append(label)
        groups.append(s.patient_id)

    if len(y) % 50 == 0:
        print("Samples:", len(y))

Xct = np.stack(Xct)[:,None,:,:,:]
Xs  = np.stack(Xs)
y   = np.array(y)
groups = np.array(groups)

print("Total samples:", len(y))
print("Class counts:", np.bincount(y))

# ============================
# SPLIT
# ============================
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr, va = next(gss.split(Xct, y, groups))

Xct_tr, Xct_va = Xct[tr], Xct[va]
Xs_tr,  Xs_va  = Xs[tr],  Xs[va]
y_tr,   y_va   = y[tr],   y[va]

# ============================
# DATASET
# ============================
class MultiDS(Dataset):
    def __init__(self,Xct,Xs,y):
        self.Xct=torch.from_numpy(Xct)
        self.Xs=torch.from_numpy(Xs)
        self.y=torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return self.Xct[i], self.Xs[i], self.y[i]

tr_loader = DataLoader(MultiDS(Xct_tr,Xs_tr,y_tr),
                       batch_size=BATCH_SIZE_STRUCT, shuffle=True)
va_loader = DataLoader(MultiDS(Xct_va,Xs_va,y_va),
                       batch_size=BATCH_SIZE_CT)

# ============================
# TRAIN STRUCTURED MLP
# ============================
counts = np.bincount(y_tr)
w = (1/counts) / (1/counts).sum() * 2
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32).to(DEVICE))
opt = torch.optim.Adam(s_model.parameters(), lr=LR)

print("\nTraining structured MLP")
best_auc = -1

for ep in range(EPOCHS_STRUCT):
    s_model.train()
    for _, xs, yb in tr_loader:
        xs,yb = xs.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(s_model(xs), yb)
        loss.backward()
        opt.step()

    s_model.eval()
    probs=[]
    with torch.no_grad():
        for _, xs, _ in va_loader:
            probs.extend(softmax_p1(s_model(xs.to(DEVICE))).cpu().numpy())
    auc = roc_auc_score(y_va, probs)
    print(f"Epoch {ep+1:02d} | val AUC={auc:.3f}")
    if auc > best_auc:
        best_auc = auc
        best_state = {k:v.cpu() for k,v in s_model.state_dict().items()}

s_model.load_state_dict(best_state)

# ============================
# LATE FUSION
# ============================
p_ct, p_s = [], []
with torch.no_grad():
    for xct,xs,_ in va_loader:
        p_ct.extend(softmax_p1(ct_model(xct.to(DEVICE))).cpu().numpy())
        p_s.extend(softmax_p1(s_model(xs.to(DEVICE))).cpu().numpy())

p_ct = np.array(p_ct)
p_s  = np.array(p_s)

best_auc, best_a = -1, 0.5
for a in np.linspace(0,1,51):
    auc = roc_auc_score(y_va, a*p_ct + (1-a)*p_s)
    if auc > best_auc:
        best_auc, best_a = auc, a

print("\nFUSION RESULTS")
print("Best alpha:", best_a)
print("Best fused AUC:", best_auc)

# ============================
# SAVE ONE COMPARISON IMAGE
# ============================
i = 0
x = Xct_va[i,0]
mid = 32

plt.figure(figsize=(11,4))
plt.subplot(1,3,1); plt.imshow(x[mid], cmap="gray"); plt.title("Axial"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(x[:,mid], cmap="gray"); plt.title("Coronal"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(x[:,:,mid], cmap="gray"); plt.title("Sagittal"); plt.axis("off")

plt.suptitle(
    f"GT={y_va[i]} | CT={p_ct[i]:.3f} | Struct={p_s[i]:.3f} | Fused={(best_a*p_ct[i]+(1-best_a)*p_s[i]):.3f}"
)

plt.savefig(os.path.join(OUTDIR,"fusion_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()

print("✅ Saved fusion_comparison.png")
