# Task 2: Object Detection on Video using YOLOv8 — My Steps

## What I did
Extended Task 1's object detection to work on video instead of a single image — 
ran YOLOv8 on a video file, got an annotated output video with detections drawn 
on every frame, and pushed it to GitHub.

---

## Step 1: Get a test video

Copied my own video into the project folder:
```bash
cp ~/Downloads/fun.MOV .
```

Checked it landed correctly:
```bash
ls
```

**Note:** YOLO/OpenCV handles `.MOV` files fine — no need to convert to `.mp4` first.

---

## Step 2: Make sure environment is activated

```bash
source .venv/bin/activate
```
Confirmed `(.venv)` shows at the start of the prompt.

---

## Step 3: Write video detection script (`detect_video.py`)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("fun.MOV", save=True)

print("Done! Check the runs/detect/predict folder for your output video.")
```

**Key difference from image detection:**
- `save=True` handles saving the entire annotated video automatically
- Don't use `results[0].show()` or `results[0].save()` like with images — for video, 
  `results` is a list of results *per frame*, so `results[0]` is just the first frame, 
  not the whole video

---

## Step 4: Run it

```bash
python detect_video.py
```
YOLO processes the video frame-by-frame, running detection on each one.

---

## Step 5: Find and view the output

```bash
ls runs/detect/predict/
```
Found `fun.mp4` — already saved in `.mp4` format (no conversion to `.avi`→`.mp4` needed).

Viewed it:
```bash
open runs/detect/predict/fun.mp4
```
Confirmed bounding boxes were drawn correctly across the video frames.

---

## Step 6: Organize into task2 folder and push to GitHub

```bash
mkdir task2
cp detect_video.py task2/
cp fun.MOV task2/
cp runs/detect/predict/fun.mp4 task2/task2_output.mp4
```

Committed and pushed:
```bash
git add task2
git commit -m "Add Task 2: YOLOv8 object detection on video"
git push
```

**Gotcha:** push got rejected because GitHub had changes I didn't have locally 
(divergent branches). Fixed by:
```bash
git config pull.rebase false
git pull
git push
```

---

## What I learned
- Video detection works frame-by-frame under the hood — same YOLOv8n model as images, 
  just applied repeatedly across every frame
- `save=True` is the easiest way to get a full annotated output video (vs. manually 
  saving individual frames)
- How to resolve a git "divergent branches" push rejection using pull + merge
- Video files take noticeably longer to process than a single image, scaling with 
  video length and resolution
