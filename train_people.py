import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "frames"))
OUT_DIR = Path(os.getenv("OUT_DIR", BASE_DIR / "imx500_people"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH = 16
EPOCHS = 8
LR = 1e-3
DEVICE = os.getenv("DEVICE", "cpu")

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = datasets.ImageFolder(DATA_DIR, transform=tfm)
class_names = ds.classes
num_classes = len(class_names)
assert num_classes >= 2, f"Need 2+ classes; found {num_classes}: {class_names}"
print("Classes:", class_names)

n_val = max(1, int(0.15 * len(ds)))
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val])

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()


def eval_acc():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x).argmax(dim=1)
            correct += (p == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


best = 0.0
for ep in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    acc = eval_acc()
    print(f"epoch {ep}/{EPOCHS} val_acc={acc:.3f}")
    if acc > best:
        best = acc
        torch.save({"model": model.state_dict(), "classes": class_names}, OUT_DIR / "people.pt")

print("Best val_acc:", best)

# Export ONNX (float model)
model.load_state_dict(torch.load(OUT_DIR / "people.pt", map_location="cpu")["model"])
model.eval().to("cpu")

dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device="cpu")
onnx_path = OUT_DIR / "people_fp32.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=18,
    input_names=["input"],
    output_names=["logits"],
    do_constant_folding=True,
)
print(f"Saved ONNX to {onnx_path}")
