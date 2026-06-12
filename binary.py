import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)
import wandb
from tqdm import tqdm
import torch.nn.functional as F

df = pd.read_csv("/scratch/anevarela/ham10000/HAM10000_metadata.csv")

LABEL_MAP = {
    'mel':   1,  # malignant
    'bcc':   1,  # malignant
    'akiec': 1,  # pre-malignant
    'nv':    0,  # benign
    'bkl':   0,  # benign
    'df':    0,  # benign
    'vasc':  0,  # benign
}

df['label'] = df['dx'].map(LABEL_MAP)
print(df['label'].value_counts())

lesions = df['lesion_id'].unique() # split by PATIENT
train_lesions, temp_lesions = train_test_split(lesions.to_numpy(), test_size=0.2, random_state=42)
val_lesions, test_lesions   = train_test_split(temp_lesions, test_size=0.5, random_state=42)

train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
val_df   = df[df['lesion_id'].isin(val_lesions)].reset_index(drop=True)
test_df  = df[df['lesion_id'].isin(test_lesions)].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

IMAGE_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # resize to fit visual transformer
    transforms.RandomHorizontalFlip(), # random transforms to include all kind of angles
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02), # color noise
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (common practice)
                         std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (common practice)
                         std=[0.229, 0.224, 0.225]),
])

class HAMDataset(Dataset):
    def __init__(self, dataframe, image_dirs, transform=None):
        self.df = dataframe
        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):
        for folder in self.image_dirs:
            path = os.path.join(folder, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image not found: {image_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self._find_image(row['image_id'])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "labels": row['label']}


IMAGE_DIRS = [
    "/scratch/anevarela/ham10000/ham10000_images_part_1",
    "/scratch/anevarela/ham10000/ham10000_images_part_2",
]

train_dataset = HAMDataset(train_df, IMAGE_DIRS, transform=train_transforms)
val_dataset   = HAMDataset(val_df,   IMAGE_DIRS, transform=val_test_transforms)
test_dataset  = HAMDataset(test_df,  IMAGE_DIRS, transform=val_test_transforms)

class_counts = train_df['label'].value_counts().to_dict()  # {0: N_benign, 1: N_malignant}
sample_weights = train_df['label'].map(
    lambda lbl: 1.0 / class_counts[lbl]
).values

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,   num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False,   num_workers=4)

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=2,
    id2label={0: "benign", 1: "malignant"},
    label2id={"benign": 0, "malignant": 1},
    ignore_mismatched_sizes=True,  # t replace classifier head
)

model = model.to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

n_benign    = (train_df['label'] == 0).sum()
n_malignant = (train_df['label'] == 1).sum()
n_total     = len(train_df)

class_weights = torch.tensor([
    n_total / (2 * n_benign),
    n_total / (2 * n_malignant),
], dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights) # weighted loss

EPOCHS      = 10
LR          = 2e-5
WARMUP_RATIO = 0.1

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

def compute_metrics(labels, preds, probs):
    return {
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "auc":          roc_auc_score(labels, probs[:, 1]),
        "f1_malignant": f1_score(labels, preds, pos_label=1),
        "sensitivity":  confusion_matrix(labels, preds, normalize='true')[1, 1],  # recall for malignant
        "specificity":  confusion_matrix(labels, preds, normalize='true')[0, 0],  # recall for benign
    }

wandb.init(project="ham10000-vit", config={
    "model": MODEL_CHECKPOINT,
    "epochs": EPOCHS,
    "lr": LR,
    "batch_size": 32,
})

best_auc = 0.0

for epoch in range(EPOCHS):

    # ── Train ──────────────────────────────────────────────
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
        loss    = criterion(outputs.logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ── Validate ───────────────────────────────────────────
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values)
            probs   = F.softmax(outputs.logits, dim=-1)
            preds   = torch.argmax(probs, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )

    print(f"\nEpoch {epoch+1} | loss: {avg_train_loss:.4f} | "
          f"AUC: {metrics['auc']:.4f} | "
          f"BalAcc: {metrics['balanced_acc']:.4f} | "
          f"Sensitivity: {metrics['sensitivity']:.4f}")

    wandb.log({"train_loss": avg_train_loss, "epoch": epoch+1, **metrics})

    # ── Checkpoint best model ──────────────────────────────
    if metrics['auc'] > best_auc:
        best_auc = metrics['auc']
        torch.save(model.state_dict(), "/scratch/anevarela/med_trial/best_model.pt")
        print(f"  ✓ New best AUC: {best_auc:.4f} — model saved")

wandb.finish()

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from PIL import Image
# Load best checkpoint
model.load_state_dict(torch.load("/scratch/anevarela/med_trial/best_model.pt", map_location=DEVICE))
model.eval()

all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating test set"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        labels       = batch["labels"].to(DEVICE)

        outputs = model(pixel_values=pixel_values)
        probs   = F.softmax(outputs.logits, dim=-1)
        preds   = torch.argmax(probs, dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

print(classification_report(
    all_labels, all_preds,
    target_names=["Benign", "Malignant/Pre-malignant"]
))

print(f"Balanced Accuracy : {balanced_accuracy_score(all_labels, all_preds):.4f}")
print(f"ROC AUC           : {roc_auc_score(all_labels, all_probs[:, 1]):.4f}")

cm = confusion_matrix(all_labels, all_preds, normalize='true')

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt=".2%", cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    ax=ax,
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Normalized Confusion Matrix — Test Set", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ViT (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — Benign vs Malignant", fontsize=13)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)

print(f"{'Threshold':>10} {'Sensitivity':>13} {'Specificity':>13} {'F1 Malignant':>14}")
print("-" * 55)

for thresh in np.arange(0.2, 0.8, 0.05):
    preds_t = (all_probs[:, 1] >= thresh).astype(int)
    cm_t    = confusion_matrix(all_labels, preds_t, normalize='true')
    tn, fp, fn, tp = confusion_matrix(all_labels, preds_t).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1   = f1_score(all_labels, preds_t, pos_label=1, zero_division=0)
    print(f"{thresh:>10.2f} {sens:>13.4f} {spec:>13.4f} {f1:>14.4f}")

# Find false negatives (missed malignancies) — the costly errors
fn_indices = np.where((all_labels == 1) & (all_preds == 0))[0]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, idx in enumerate(fn_indices[:10]):
    row      = test_df.iloc[idx]
    img_path = next(
        os.path.join(d, f"{row['image_id']}.jpg")
        for d in IMAGE_DIRS
        if os.path.exists(os.path.join(d, f"{row['image_id']}.jpg"))
    )
    img  = Image.open(img_path)
    conf = all_probs[idx, 1]

    axes[i].imshow(img)
    axes[i].set_title(f"True: {row['dx']}\nConf: {conf:.2f}", fontsize=8)
    axes[i].axis("off")

plt.suptitle("False Negatives — Missed Malignancies", fontsize=13, color="firebrick")
plt.tight_layout()
plt.savefig("false_negatives.png", dpi=150)
