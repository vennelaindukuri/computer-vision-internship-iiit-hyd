# Task 3: Custom YOLOv8 Training on Cats vs Dogs Dataset — My Steps

## What I did
Trained a custom YOLOv8 model from scratch on a labeled Cats vs Dogs dataset (instead 
of just using the pretrained COCO model), evaluated it with real metrics (precision, 
recall, mAP), tested it on a new image, and pushed everything to GitHub.

---

## Step 1: Get a labeled dataset from Roboflow

Found a pre-labeled "Cats vs Dogs Computer Vision Dataset" on Roboflow Universe 
(82 images, 2 classes: cat, dog, object detection format).

Installed the roboflow package:
```bash
pip install roboflow
```

Created `download_dataset.py`:
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("zaka-ai-challenge").project("cats-vs-dogs-mpr6n")
version = project.version(1)
dataset = version.download("yolov8")
```

Ran it:
```bash
python download_dataset.py
```
This created a `Cats-vs-Dogs-1/` folder with `train/images`, `train/labels`, and `data.yaml`.

---

## Step 2: Fix a broken data.yaml

**Gotcha:** the downloaded `data.yaml` referenced `valid/images` and `test/images` 
folders that didn't actually exist — the dataset only had a `train` split.

Fixed it by pointing all three paths to the same train folder:
```yaml
test: ../train/images
train: ../train/images
val: ../train/images
```
(Not ideal practice for a real project, but a reasonable workaround for a small demo dataset.)

---

## Step 3: Train the model

Created `train_model.py`:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="Cats-vs-Dogs-1/data.yaml", epochs=50, imgsz=640)
```

Ran it:
```bash
python train_model.py
```
Took a while on CPU (no GPU on MacBook Air) — let it run through all 50 epochs 
without interrupting.

---

## Step 4: Evaluate the model — get real metrics

Created `evaluate_model.py`:
```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="Cats-vs-Dogs-1/data.yaml")

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
```

Ran it:
```bash
python evaluate_model.py
```

**Results:**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Cat | 0.996 | 1.0 | 0.995 | 0.859 |
| Dog | 0.971 | 1.0 | 0.995 | 0.899 |
| **All** | **0.983** | **1.0** | **0.995** | **0.879** |

Also generated visual outputs in `runs/detect/val/`:
- `confusion_matrix.png` — predicted vs actual class breakdown
- `results.png` (from training) — loss curves and metrics over epochs
- `val_batch*_pred.jpg` — sample validation images with predicted boxes

**Important caveat:** since train and validation used the same folder (no separate 
val split existed), these metrics are optimistic — the model was evaluated on data 
it already learned from, so this isn't a true test of generalization to unseen data.

---

## Step 5: Test on a brand new image

Created `test_trained_model.py`:
```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("new_pet_photo.jpg")

results[0].show()
results[0].save("task3_output.jpg")
```

Ran it on a photo the model had never seen — correctly detected the cat/dog.

---

## Step 6: Organize into task3 folder and push to GitHub

```bash
mkdir task3
cp download_dataset.py task3/
cp train_model.py task3/
cp evaluate_model.py task3/
cp test_trained_model.py task3/
cp task3_output.jpg task3/
cp runs/detect/train/weights/best.pt task3/
cp runs/detect/val/confusion_matrix.png task3/
cp runs/detect/train/results.png task3/
```

Committed and pushed:
```bash
git add task3
git commit -m "Add Task 3: Custom YOLOv8 training on Cats vs Dogs dataset"
git push
```

**Gotcha (same as Task 2):** push got rejected due to divergent branches. Fixed with:
```bash
git pull
git push
```
When a Vim merge commit message editor popped up, saved and exited with `Esc` then `:wq` then Enter.

---

## What I learned
- Difference between using a pretrained model (Tasks 1 & 2) vs. training a custom 
  model on labeled data (Task 3)
- How to source and download a labeled dataset via Roboflow
- What `data.yaml` does and why train/val/test paths matter
- How to actually train a YOLOv8 model with `model.train()`
- What precision, recall, mAP50, and mAP50-95 mean and how to interpret them
- The importance of a proper train/val split — and the limitation of reusing 
  the same data for both
- Reading a confusion matrix to see class-level performance
- Resolving git merge conflicts using Vim's `:wq` to save and exit 
