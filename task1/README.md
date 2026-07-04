# Task 1: Object Detection using YOLOv8 — My Steps

## What I did
Set up a Python environment on my Mac, installed Ultralytics YOLOv8, ran object detection 
on sample images and my own photos, and pushed everything to GitHub.

---

## Step 1: Set up virtual environment

Checked Python version first:
```bash
python3 --version
```

Created and activated a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
(Prompt changes to show `(.venv)` at the start when active.)

**Note:** On Mac, `python` and `pip` don't work by default — had to use `python3`. 
Once the venv is activated, `python` and `pip` work fine inside it.

---

## Step 2: Install Ultralytics

```bash
pip install ultralytics
```
This installs YOLOv8 along with PyTorch, OpenCV, NumPy, and other dependencies.

Verified install worked:
```bash
python -c "from ultralytics import YOLO; print('success')"
```

---

## Step 3: Get a test image

Downloaded a sample image (bus + people) to test with:
```bash
curl -L -o test.jpg https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
```
**Gotcha:** first tried without `-L` (follow redirects) and it saved a broken 15-byte file 
instead of the actual image. Always check file size after downloading:
```bash
ls -la test.jpg
```

---

## Step 4: Write detection script (`detect.py`)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("test.jpg")

results[0].show()
results[0].save("output.jpg")
```

Ran it:
```bash
python detect.py
```
First run auto-downloads `yolov8n.pt` (pretrained model weights, ~6MB).

---

## Step 5: Test on my own photos

Copied my own images into the project folder:
```bash
cp ~/Downloads/dogcar.jpg .
```
Checked filename/extension matched exactly (`.jpg` vs `.jpeg` matters), then updated 
`detect.py` to point to the new filename and re-ran it.

**Results I got:**
- `test.jpg` → bus + multiple persons detected
- `dogwalk.jpeg` → 3 persons, 2 dogs
- `dogcar.jpg` → 1 truck, 1 teddy bear

---

## Step 6: Push to GitHub

Cloned my internship repo:
```bash
cd ~/Desktop
git clone https://github.com/vennelaindukuri/computer-vision-internship-iiit-hyd.git
cd computer-vision-internship-iiit-hyd
```

Made a task1 folder and copied my files in:
```bash
mkdir task1
cp /path/to/detect.py task1/
cp /path/to/test.jpg task1/
cp /path/to/output.jpg task1/
```

Committed and pushed:
```bash
git add task1
git commit -m "Add Task 1: YOLOv8 object detection"
git push
```

**Gotcha:** tried chaining `git add` and `git commit -m "..."` on one line — git threw 
an "unknown switch" error. They need to be run as separate commands.

---

## What I learned
- Setting up and activating a Python virtual environment
- Installing and running a pretrained YOLOv8 model for object detection
- Reading detection output (class names, confidence, bounding boxes, inference speed)
- YOLOv8n is trained on COCO's 80 classes — anything outside that won't be detected
- Basic git workflow: clone → add → commit → push
