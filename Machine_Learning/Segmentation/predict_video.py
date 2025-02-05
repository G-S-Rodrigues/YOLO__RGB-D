from ultralytics import YOLO
import cv2

# Load a pretrained YOLO11n model
model = YOLO("/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/r4f_yolo11x_seg.pt",
             task="segment")

model.predict("/media/r4f-orin/SD/Data/vinha-09-04/videos/row1.mp4",
              save=True,
            #   batch=32,
              conf=0.4, 
              device=0, 
              project="/media/r4f-orin/SD/Resutls/vinha-09-04/")

#
#ffmpeg -i run1_camera_f.avi -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 192k run1_camera_f.mp4


