import os
from ultralytics import YOLO
import cv2
import numpy as np
import yaml

def get_inp(yaml_name): # 'predict_dir.yaml'
    with open(yaml_name, 'r') as file:
        predict_dir = yaml.safe_load(file)

    file_path = os.path.join(predict_dir["inp_pth"], predict_dir["inp_file_name"])
    file_path_out = os.path.join(predict_dir["inp_pth"], predict_dir["out_file_name"])
    model_path = predict_dir["model_path"]
    threshold = float(predict_dir["threshold"])
    
    return file_path, file_path_out, model_path, threshold

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
        


def main(args=None):
    file_path, file_path_out, model_path , threshold = get_inp('predict_dir.yaml')
    
    # Load the model
    model = YOLO(model_path)
    class_names = model.names
    print('Class Names: ', class_names)
    colors = [(0,0,255), (255,0,0), (0,255,255)]

    cap = cv2.VideoCapture(file_path)
    _ , img = cap.read()
    h, w, _ = img.shape
    out = cv2.VideoWriter(file_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))
    
    while True:
        success, img = cap.read()
        if not success:
            break

        
        results = model.predict(img, stream=True, conf=threshold)
        
        
        # print(results)
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):

                seg = cv2.resize(seg, (w, h))
                img = overlay(img, seg, colors[int(box.cls)], 0.5)
            
                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])
            
                plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names[int(box.cls)]} {float(box.conf):.1}',line_thickness=1)
        
        out.write(img)
        # cv2.imshow('img', img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()