import os
from pathlib import Path

import model_compression_toolkit as mct
import torch
import torch.onnx as torch_onnx
from edgemdt_tpc import get_target_platform_capabilities
from torch import nn
from torchvision import datasets, models, transforms

# Work around torch 2.9 defaulting to dynamo=True in torch.onnx.export, which
# clashes with MCT's dynamic_axes usage. Force legacy exporter.
_orig_export = torch_onnx.export
def _export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _orig_export(*args, **kwargs)
torch_onnx.export = _export

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "frames"))
OUT_DIR = Path(os.getenv("OUT_DIR", BASE_DIR / "imx500_people"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
REP_SAMPLES = 200           # how many calibration samples to use
REP_BATCH = 8               # >1 is faster than 1
DEVICE = os.getenv("DEVICE", "cpu")

# Use deterministic-ish transforms for representative data (no jitter/flip)
rep_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = datasets.ImageFolder(DATA_DIR, transform=rep_tfm)
class_names = ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

# Rebuild model architecture and load your trained weights
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

ckpt = torch.load(OUT_DIR / "people.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)


# Representative dataset generator (yields batches of input tensors)
def representative_data_gen():
    # simple sequential sampling; you can randomize if you want
    n = min(len(ds), REP_SAMPLES)
    i = 0
    while i < n:
        batch = []
        for _ in range(REP_BATCH):
            if i >= n:
                break
            x, _ = ds[i]
            batch.append(x)
            i += 1
        if not batch:
            break
        yield [torch.stack(batch, dim=0)]  # list/tuple of model inputs


# Get IMX500 target platform capabilities (TPC)
tpc = get_target_platform_capabilities(tpc_version="1.0", device_type="imx500")

# Quantize (PTQ) with MCT
# Note: the exact keyword for passing TPC can differ across MCT versions,
# so we try the common one first, then a fallback.
try:
    quant_model, quant_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_data_gen,
        target_platform_capabilities=tpc,
    )
except TypeError:
    # Some variants use different keyword names; TPC repo shows an example,
    # but packaging can differ. This fallback is worth trying.
    quant_model, quant_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_data_gen,
        target_resource_utilization=tpc,
    )

# Export quantized ONNX using MCT exporter
onnx_q_path = OUT_DIR / "people_int8_mct.onnx"
mct.exporter.pytorch_export_model(
    model=quant_model,
    save_model_path=str(onnx_q_path),
    repr_dataset=representative_data_gen,   # correct arg name for exporter
    onnx_opset_version=18,
)

print(f"Saved quantized ONNX: {onnx_q_path}")
