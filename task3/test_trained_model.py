from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("puppy.jpg")

results[0].show()
results[0].save("task3_output.jpg")
