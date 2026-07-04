from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("fun.MOV", save=True)

print("Done! Check the runs/detect/predict folder for your output video.")
