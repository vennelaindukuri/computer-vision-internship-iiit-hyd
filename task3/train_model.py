from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="Cats-vs-Dogs-1/data.yaml", epochs=50, imgsz=640)
