## Internship Summary

This repository documents my computer vision internship work at IIIT Hyderabad, 
using YOLOv8 (Ultralytics) for object detection.

### Task 1: Object Detection on Images
Used a pretrained YOLOv8n model to detect multiple objects, people, and animals 
in images. Covers environment setup, running inference, and interpreting detection 
output (classes, confidence scores, bounding boxes).

### Task 2: Object Detection on Video
Extended the same pretrained model to run detection frame-by-frame on video files, 
producing an annotated output video with bounding boxes drawn across every frame.

### Task 3: Custom Model Training (Cats vs Dogs)
Trained a custom YOLOv8 model from scratch on a labeled Cats vs Dogs dataset, 
moving beyond pretrained detection into actual model training and evaluation. 
Achieved strong results: **Precision: 0.983, Recall: 1.0, mAP50: 0.995, 
mAP50-95: 0.879**. Learned to interpret confusion matrices, precision-recall 
curves, and training loss curves.

---

## Next Steps: Medical Imaging Focus

For the remainder of this internship, I've chosen to focus on **applying 
computer vision to the medical field** — specifically building a **pneumonia 
detection model** using chest X-ray images. This direction lets me explore 
object detection and classification techniques in a high-impact, real-world 
domain, building on the YOLOv8 foundations from Tasks 1-3.
