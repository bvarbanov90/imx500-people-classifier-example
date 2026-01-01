# IMX500 People Classifier (Personal Playground)

End-to-end example for capturing your own frames, training, quantising, packaging, and running a custom IMX500 classification model on Raspberry Pi 5. Uses a simple MobileNetV3-Small finetune on a folder-per-person dataset. This is a personal, hands-on project to learn the ins and outs of practical AI/ML vision on-device and expect a few cheerful notes and experiments along the way. Official docs reference: https://www.raspberrypi.com/documentation/accessories/ai-camera.html

## Layout
- `train_people.py` – finetune MobileNetV3 on your dataset and export FP32 ONNX.
- `quantize_people_mct.py` – PTQ int8 export for IMX500 using MCT + TPC.
- `picamera2/` – Picamera2 with IMX500 examples (editable install).
- `imx500_people/` – model outputs; RPKs get written here.
- `frames/` – **placeholder for your dataset** (`frames/<person_name>/*.jpg`). Not tracked. Bring your own smiles.

### External dependencies (system/SDK)
- Raspberry Pi AI Camera tools (`imx500-package`, `imx500_converter`, postconverter/packager) per the official docs above.
- libcamera + IMX500 firmware (from Pi OS).
- System build deps: `libcap-dev`, unzip, and typical build tools.

### Related personal projects (for inspiration)
- GPU pipeline tinkering: https://github.com/bvarbanov90/basic-gpu-pipeline-example
- Misc. experiments: https://github.com/bvarbanov90/octopus-energy-tariff-notifier
- Profile hub: https://github.com/bvarbanov90

## Prereqs
- Raspberry Pi 5 + IMX500 AI Camera, latest Pi OS with libcamera working.
- IMX500 tooling (`imx500-package`, `imx500_converter` etc.) installed from the AI Camera docs.
- System packages for build: `sudo apt-get install libcap-dev`.
- Python 3.11/3.13 venv with system site packages (needed for libcamera bindings).

## Setup
```bash
# From repo root
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e picamera2          # use local picamera2
```

## Data
- Create `frames/` with one subfolder per person/class, containing images. You can capture frames with Picamera2 scripts or any camera app, then drop them into `frames/<name>/`.
- Example: `frames/boris/*.jpg`, `frames/eva/*.jpg`, `frames/tsetso/*.jpg`.
- Images are ignored by git; these folders are placeholders only.

## Personal workflow snapshot
- Capture frames of friends/family into `frames/<name>/`.
- Fine-tune MobileNetV3 with `train_people.py`.
- PTQ int8 with `quantize_people_mct.py`.
- Package to `.rpk` via `imx500-package.sh`.
- Run live classification on the IMX500 and enjoy the on-device magic.

## Train
```bash
source .venv/bin/activate
python train_people.py
# outputs: imx500_people/people.pt and imx500_people/people_fp32.onnx
```

## Quantise (int8)
```bash
source .venv/bin/activate
python quantize_people_mct.py
# outputs: imx500_people/people_int8_mct.onnx
```

## Package to RPK (with official IMX500 packager)
```bash
# Convert ONNX -> packerOut.zip
source imx500-py311/bin/activate   # or your imx500 converter env
python -m imx500_converter.main_pt -i imx500_people/people_int8_mct.onnx \
  -o imx500_people/conv_out_clean --overwrite-output

# Add required metadata and build .rpk
/usr/lib/imx500-tools/packager/imx500-package.sh \
  -i imx500_people/conv_out_clean/packerOut.zip \
  -o imx500_people/conv_pkg_clean
# result: imx500_people/conv_pkg_clean/network.rpk
```

## Run on camera
```bash
source imx500-py313/bin/activate   # system-site venv with libcamera
python picamera2/examples/imx500/imx500_classification_demo.py \
  --model imx500_people/conv_pkg_clean/network.rpk \
  --labels imx500_people/labels.txt
```

## Notes
- `frames/` is intentionally ignored; add your own images.
- If you need a headless run, set `show_preview=False` inside the demo before running.
- The demo patch allows small label sets (3 classes) and guards softmax/top-k safely.
- This lives on my personal GitHub (https://github.com/bvarbanov90) as a learning sandbox so feel free to fork, poke, and have fun. Keep it light-hearted; the goal is to tinker and learn.
