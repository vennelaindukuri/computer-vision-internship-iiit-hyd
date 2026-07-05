from roboflow import Roboflow

rf = Roboflow(api_key="Lsfk1ZAy1t0eeo0UU9P4")
project = rf.workspace("zaka-ai-challenge").project("cats-vs-dogs-mpr6n")
version = project.version(1)
dataset = version.download("yolov8")
