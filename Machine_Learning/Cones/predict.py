import os
from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from PIL import Image

def main(args=None):
    name = "m_density2"
    file_path = f"/media/r4f-orin/SD/Data/lab/videos/{name}.mp4"
    base_path = f"/media/r4f-orin/SD/Data/lab/videos/"   
    base_model = f"/media/r4f-orin/SD/dev/Neural_Network_Yolo/YOLO_models/Cones"
    model_name = f"cones_r4f.pt"
    threshold = 0.4
    

    model_path = os.path.join(base_model, model_name)  
    
    # Clean the model name by removing the "r4f_" prefix and the extension
    model_name_clean = model_name
    if model_name_clean.startswith("r4f_"):
        model_name_clean = model_name_clean[len("r4f_"):]
    model_name_clean = model_name_clean.split('.')[0]  # Take text before the first "."

    file_name = f"predict_{name}_{model_name_clean}.mp4"
    file_path_out = os.path.join(base_path, file_name) 
    
    # Check if the file exists and modify the name if necessary
    counter = 1
    while os.path.exists(file_path_out):
        file_name = f"predict_{name}_{model_name_clean}_{counter}.mp4"
        file_path_out = os.path.join(base_path, file_name)
        counter += 1

    # Load the model
    model = YOLO(model_path, task="segment")
    # print(model)

    # detection_model = YOLO(model_path)
    # model = YOLO("yolo11n-seg.pt")  # Load a segmentation model
    # model.model.load_state_dict(detection_model.model.state_dict(), strict=False)
    # model.save("segmentation_model.pt")

    class_names = model.names
    print('Class Names: ', class_names)

    # colors = [(255,0,0), (0,255,0), (0,0,255)]

    cap = cv2.VideoCapture(file_path)
    _ , img = cap.read()
    h, w, _ = img.shape
    out = cv2.VideoWriter(file_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))
    cap.release()
    
    results = model.predict(file_path, stream=True, device=0, conf=threshold)
    
    for i, r in enumerate(results):
        # Plot results image
        img = r.plot(color_mode="class",
                     line_width= 2,
                     font_size= 10
                    ) 
        # out.write(img)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # cap.release()
    out.release()
    cv2.destroyAllWindows()

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
            

def plot_one_box(x, img, color=None, label=None, line_thickness=10):
    # Plots one bounding box on image img
    tl = line_thickness  # line/font thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, tl, cv2.LINE_AA) 
        cv2.putText(img, label, c1, cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        

if __name__ == '__main__':
    main()