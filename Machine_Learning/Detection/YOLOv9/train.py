#!./.venv
from ultralytics import YOLO
import os

data_path = 'data.yaml'
model_path = '/home/guilh/Vineyard_autonomous_robot/DL_Trunk_Detection/YOLOv9/runs/train2/weights/best.pt'
n_epochs = 10
bs = -1 # 16
gpu_id = 0
verbose = True
validate = True
resume = False
patience = 25 

# Specify the save directory for training runs
save_dir = '/home/guilh/Vineyard_autonomous_robot/DL_Trunk_Detection/YOLOv9/runs'
os.makedirs(save_dir, exist_ok=True)

# Load and train the model
model = YOLO("yolov9c.pt")  # build a new model from scratch
#model = YOLO(model_path)
#results = model.train(
#    data=data_path,
#    epochs=n_epochs,
#    batch=bs,
#    device=gpu_id,
#    verbose=verbose,
#    val=validate,
#    project=save_dir,
#    resume=resume,
#    patience = patience
#)
