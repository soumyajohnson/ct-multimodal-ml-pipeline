import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# =========================================================
# SETTINGS
# =========================================================
CACHE_PATH = r"your_cache_path_here"  # <-- path to the CT cache built in build_cache.py
# Example (Windows): r"E:\lidc_cache\lidc_ct_cache_iso_fp16_500.npz"

OUTDIR = "outputs_train"
os.makedirs(OUTDIR, exist_ok=True)

EPOCHS = 20
BATCH_SIZE = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("✅ Using device:", DEVICE)

# =========================================================
# LOAD CACHE
# =========================================================
data = np.load(CACHE_PATH, allow_pickle=True)
X = data["Xct"].astype(np.float32)  # cast back to fp32 for torch
y = data["y"].astype(np.int64)
pids = data["groups"]

print("Loaded:", X.shape, X.dtype, "| y counts:", np.bincount(y))

# =========================================================
# SPLIT (patient-wise)
# =========================================================
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=pids))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("Train counts:", np.bincount(y_train))
print("Val counts  :", np.bincount(y_val))

# =========================================================
# Dataset / Loader
# =========================================================
class CropDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# On Windows, if DataLoader hangs, set num_workers=0
NUM_WORKERS = 0

train_loader = DataLoader(
    CropDS(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    CropDS(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =========================================================
# 3D ResNet-18 (pure PyTorch)
# =========================================================
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
        out = F.relu(out)
        return out

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

def resnet3d18(num_classes=2):
    return ResNet3D(BasicBlock3D, [2,2,2,2], num_classes=num_classes, in_channels=1)

model = resnet3d18(num_classes=2).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# weighted loss (use ONE balancing strategy)
counts = np.bincount(y_train)
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum() * 2.0
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))

print("Loss class weights:", weights)

# =========================================================
# Metrics
# =========================================================
def evaluate(loader):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            p1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            pred = logits.argmax(1).detach().cpu().numpy()

            ys.extend(yb.detach().cpu().numpy().tolist())
            preds.extend(pred.tolist())
            probs.extend(p1.tolist())

    ys = np.array(ys)
    preds = np.array(preds)
    probs = np.array(probs)

    acc = (preds == ys).mean()
    f1 = f1_score(ys, preds, zero_division=0)
    try:
        auc = roc_auc_score(ys, probs)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(ys, preds, labels=[0, 1])
    return acc, f1, auc, cm

# =========================================================
# Train
# =========================================================
print("\nStarting training from SSD cache...")
best_auc = -1.0

for ep in range(1, EPOCHS + 1):
    model.train()
    running = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

        running += loss.item() * yb.size(0)

    train_acc, train_f1, train_auc, _ = evaluate(train_loader)
    val_acc, val_f1, val_auc, val_cm = evaluate(val_loader)

    print(
        f"Epoch {ep:02d}/{EPOCHS} | "
        f"loss={running/len(train_loader.dataset):.4f} | "
        f"train acc={train_acc:.3f} f1={train_f1:.3f} auc={train_auc:.3f} | "
        f"val acc={val_acc:.3f} f1={val_f1:.3f} auc={val_auc:.3f}"
    )
    print("Val confusion matrix [[TN FP],[FN TP]]:\n", val_cm)

    if not np.isnan(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        ckpt = os.path.join(OUTDIR, "best_resnet3d18.pt")
        torch.save(model.state_dict(), ckpt)
        print("✅ Saved best:", ckpt)

print("\nDone.")
print("Best val AUC:", best_auc)
