from ultralytics import YOLO

# Load a model
# model = YOLO("/home/guilh/Robotics4Farmers/Machine_Learning/YOLO_models/r4f_yolov8m_seg.pt")  # load a custom trained model

model = YOLO("yolo11m-seg.pt")

# Export the model
success = model.export(format="onnx", opset=11, simplify=True)  # export the model to onnx format FP16
assert success