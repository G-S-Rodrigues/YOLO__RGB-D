import os
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/r4f_yolo11x_seg.pt")
# model.names = {0: "Person", 1: "Post", 2: "Trunk"}
# model.task ='segment'
 
# Define the input folder containing images
input_folder = "/media/r4f-orin/SD/Data/Machine_Learning/R4F_seg/test/images/"  # Change this to your image folder path
output_folder = "/media/r4f-orin/SD/Data/Machine_Learning/R4F_seg/test/images/"  # Output folder for results

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Run YOLO prediction on the image
model.predict(input_folder, save=True, conf=0.25, device=0, project=output_folder)

print("Processing complete!")
