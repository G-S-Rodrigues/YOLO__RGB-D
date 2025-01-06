from ultralytics.data.annotator import auto_annotate



auto_annotate(
    data='/home/guilh//data_tese/Machine_Learning/R4f_final_seg/resized',
    det_model='/home/guilh/Robotics4Farmers/Machine_Learning/YOLO_models/r4f_yolov8m_seg.pt',
    sam_model='sam2_l.pt',
    device="cuda",
    output_dir='/home/guilh//data_tese/Machine_Learning/R4f_final_seg/lables'
)