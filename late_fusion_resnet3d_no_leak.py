import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ----------------------------
# PATHS
# ----------------------------
CT_CACHE_PATH = r"C:\Users\ragha\OneDrive\Documents\LIDC CACHE\lidc_ct_cache_iso_fp16_500.npz"
STRUCT_CACHE_PATH = r"C:\Users\ragha\OneDrive\Documents\LIDC CACHE\lidc_struct_cache_500_no_leak_2_attr.npz"
CT_CKPT = r"outputs_train\best_resnet3d18.pt"

OUTDIR = "outputs_fusion_resnet3d"
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------
# SETTINGS
# ----------------------------
SEED = 42
EPOCHS_STRUCT = 25
BATCH_SIZE_CT = 2
BATCH_SIZE_STRUCT = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# LOAD CACHES
# ----------------------------
dct = np.load(CT_CACHE_PATH, allow_pickle=True)
Xct = dct["Xct"].astype(np.float32)  # back to fp32
y = dct["y"].astype(np.int64)
groups = dct["groups"]

ds = np.load(STRUCT_CACHE_PATH, allow_pickle=True)
Xs = ds["Xs"].astype(np.float32)

# sanity
assert len(Xct) == len(Xs) == len(y) == len(groups), "Cache lengths mismatch"

print("✅ Device:", DEVICE)
print("Xct:", Xct.shape, Xct.dtype, "| Xs:", Xs.shape, "| y:", np.bincount(y))
if "attrs" in ds:
    print("Structured attrs used:", list(ds["attrs"]))

# ----------------------------
# SPLIT (patient-wise)
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr, va = next(gss.split(Xct, y, groups=groups))

Xct_tr, Xct_va = Xct[tr], Xct[va]
Xs_tr,  Xs_va  = Xs[tr],  Xs[va]
y_tr,   y_va   = y[tr],   y[va]

print("Train counts:", np.bincount(y_tr))
print("Val counts  :", np.bincount(y_va))

# ----------------------------
# DATASET
# ----------------------------
class MultiDS(Dataset):
    def __init__(self, Xct, Xs, y):
        self.Xct = torch.from_numpy(Xct)
        self.Xs  = torch.from_numpy(Xs)
        self.y   = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xct[i], self.Xs[i], self.y[i]

NUM_WORKERS = 0  # Windows-safe
tr_loader = DataLoader(MultiDS(Xct_tr, Xs_tr, y_tr), batch_size=BATCH_SIZE_STRUCT, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True)
va_loader = DataLoader(MultiDS(Xct_va, Xs_va, y_va), batch_size=BATCH_SIZE_CT, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

# ----------------------------
# ResNet3D definition (must match your training script)
# ----------------------------
class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, in_channels=1):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, stride=st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def resnet3d18(num_classes=2):
    return ResNet3D(BasicBlock3D, [2,2,2,2], num_classes=num_classes, in_channels=1)

def softmax_p1(logits):
    return torch.softmax(logits, dim=1)[:, 1]

# ----------------------------
# Load frozen CT model
# ----------------------------
ct_model = resnet3d18(num_classes=2).to(DEVICE)
ct_model.load_state_dict(torch.load(CT_CKPT, map_location=DEVICE))
ct_model.eval()
for p in ct_model.parameters():
    p.requires_grad = False

# ----------------------------
# Structured MLP
# ----------------------------
class StructMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x): return self.net(x)

s_model = StructMLP(in_dim=Xs.shape[1]).to(DEVICE)

counts = np.bincount(y_tr)
w = 1.0 / (counts + 1e-6)
w = w / w.sum() * 2.0
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(DEVICE))
opt = torch.optim.AdamW(s_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

print("\nTraining structured MLP (no-leak features)...")
best_auc = -1.0
best_state = None

for ep in range(1, EPOCHS_STRUCT + 1):
    s_model.train()
    for _, xs, yb in tr_loader:
        xs = xs.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(s_model(xs), yb)
        loss.backward()
        opt.step()

    s_model.eval()
    probs = []
    with torch.no_grad():
        for _, xs, _ in va_loader:
            xs = xs.to(DEVICE, non_blocking=True)
            probs.extend(softmax_p1(s_model(xs)).cpu().numpy().tolist())
    auc = roc_auc_score(y_va, np.array(probs))
    print(f"Epoch {ep:02d} | val AUC={auc:.3f}")
    if auc > best_auc:
        best_auc = auc
        best_state = {k: v.detach().cpu() for k, v in s_model.state_dict().items()}

s_model.load_state_dict(best_state)
print("Best structured val AUC:", best_auc)

# ----------------------------
# Late Fusion
# ----------------------------
p_ct, p_s = [], []
with torch.no_grad():
    for xct, xs, _ in va_loader:
        xct = xct.to(DEVICE, non_blocking=True)
        xs  = xs.to(DEVICE, non_blocking=True)
        p_ct.extend(softmax_p1(ct_model(xct)).cpu().numpy().tolist())
        p_s.extend(softmax_p1(s_model(xs)).cpu().numpy().tolist())

p_ct = np.array(p_ct)
p_s  = np.array(p_s)



best_auc, best_a = -1.0, 0.5
for a in np.linspace(0, 1, 51):
    fused = a * p_ct + (1 - a) * p_s
    auc = roc_auc_score(y_va, fused)
    if auc > best_auc:
        best_auc, best_a = auc, a

fused_best = best_a * p_ct + (1 - best_a) * p_s
# ----------------------------
# Pick best threshold on VAL (maximize F1)
# ----------------------------
best_f1, best_t = -1.0, 0.5
for t in np.linspace(0.05, 0.95, 19):
    pred_t = (fused_best >= t).astype(int)
    f1_t = f1_score(y_va, pred_t, zero_division=0)
    if f1_t > best_f1:
        best_f1, best_t = f1_t, t

pred = (fused_best >= best_t).astype(int)
cm = confusion_matrix(y_va, pred, labels=[0, 1])

print("\nFUSION RESULTS (ResNet3D + no-leak structured)")
print("Best alpha:", best_a)
print("Best fused AUC:", best_auc)
print("Best threshold (val, max F1):", best_t)
print("Best F1:", best_f1)
print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)

