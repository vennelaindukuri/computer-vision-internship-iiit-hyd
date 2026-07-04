from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("dogcar.jpg")

results[0].show()
results[0].save("dogoutput.jpeg")