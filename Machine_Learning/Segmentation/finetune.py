#!/usr/bin/env python
from ultralytics import YOLO

def main():
    model = YOLO("/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/r4f_yolo11n_seg.pt") 
    model.train(
        data="train_data.yaml",
        imgsz=(384, 640),
        epochs=200,
        batch=16,
        device=0,
        project="/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/fine_tune",
        name="finetune_yolo11n/train",
        pretrained=True  # re-use the pre-trained weights as initialization
    )

if __name__ == '__main__':
    main()
