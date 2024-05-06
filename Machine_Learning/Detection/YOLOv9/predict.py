import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = '/home/guilh/Data_Vineyard_autonomous_robot/Vinha_camera/videos'

video_path = os.path.join(VIDEOS_DIR, 'row1.mp4')
video_path_out = os.path.join(VIDEOS_DIR, 'row1_predict.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
print(H,W)
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = '/home/guilh/Vineyard_autonomous_robot/DL_Trunk_Detection/YOLOv9/runs/train2/weights/best.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > threshold:
            if class_id == 0.0:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()