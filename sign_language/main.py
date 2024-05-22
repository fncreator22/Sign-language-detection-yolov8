from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
result = model.train(data=r"C:\Users\Sagar\Downloads\American Sign Language Letters.v1-v1.yolov8\sign_language\data.yaml", epochs=5, imgsz=640, deterministic=True)
