from ultralytics import YOLO

# Load a model
# model = YOLO("/home/guilh/Robotics4Farmers/Machine_Learning/YOLO_models/r4f_yolov8m_seg.pt")  # load a custom trained model

model = YOLO("/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/r4f_yolo11n_seg_s.pt")

# Export the model
success = model.export(format="onnx", simplify=True, imgsz=(640,360))  # export the model to onnx format FP16
assert success