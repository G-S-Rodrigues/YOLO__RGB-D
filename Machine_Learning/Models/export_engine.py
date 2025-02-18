from ultralytics import YOLO

# Load a model
model = YOLO("/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/r4f_yolo11n_seg.pt")  # load a custom trained model

# Export the model
success = model.export(format='engine',
                        # device="dla:0",
                        # half=True,
                        # int8=True,
                        # imgsz=(384,640),
                        # batch=1,
                        # data="data.yaml",
                        )
    
assert success