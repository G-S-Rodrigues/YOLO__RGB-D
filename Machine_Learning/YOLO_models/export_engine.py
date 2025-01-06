from ultralytics import YOLO

# Load a model
model = YOLO("/home/guilh/Robotics4Farmers/Machine_Learning/YOLO_models/r4f_yolov8m_seg.pt")  # load a custom trained model

# Export the model
success = model.export(format='engine',
                        imgsz=640,
                        optimize=False,
                        half=True,
                        int8=False,
                        dynamic=False,
                        opset=11)

assert success