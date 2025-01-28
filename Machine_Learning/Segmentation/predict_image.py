from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/yolo11x/train/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("/home/r4f-orin/Downloads/FIRA6.png", save=True, conf=0.25, device=0, project="/media/r4f-orin/SD/dev/Neural_Network_Yolo/runs")