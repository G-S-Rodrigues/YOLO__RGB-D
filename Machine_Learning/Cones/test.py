
from ultralytics import YOLO
import os

base_model = f"/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/Cones"
model_name = f"best.pt"

    

model_path = os.path.join(base_model, model_name)  

model = YOLO(model_path, task="segment")

model.predict("/media/r4f-orin/SD/Data/lab/videos/m_density2.mp4",
              save=True,
            # batch=32,
              conf=0.4, 
              device=0,
              project="/media/r4f-orin/SD/Data/lab/videos"
              )