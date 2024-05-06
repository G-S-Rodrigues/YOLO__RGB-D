from ultralytics.data.annotator import auto_annotate



auto_annotate(
    data='/home/guilh/Data_Vineyard_autonomous_robot/Machine_Learning/All_images_6fps',
    det_model='/home/guilh/Vineyard_autonomous_robot/Machine_Learning/Segmentation/model/train/weights/best.pt',
    sam_model='sam_l.pt',
    device="cuda",
    output_dir='/home/guilh/Data_Vineyard_autonomous_robot/Machine_Learning/All_labels_6fps'
)