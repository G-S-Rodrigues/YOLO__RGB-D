import numpy as np
import cv2
import time
import pyrealsense2 as rs
import argparse
import os.path
from typing import List, Dict, Tuple
# Vision model
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks

from utils import BoundingBox2D, Point2D, Mask, Detection


class Yolo():
    
    def __init__(self):
        
        parser = argparse.ArgumentParser(description="Read recorded bag file and display the image with object identification.\
                                            It also obtains the coordinates of the object relative to the camera")

        # Add argument which takes path to a bag file as an input
        parser.add_argument("-i", "--input", type=str, default="/home/guilh/Data_Vineyard_autonomous_robot/Vinha-09-04/row1.bag", help="Path to the bag file")
        parser.add_argument("-y", "--yolo", type=str, default="/home/guilh/r4f_yolov8m_seg.pt", help="Path to yolo model")
        parser.add_argument("-t", "--threshold", type=float, default=0.60, help="Threshold for the predictions confidence ")
        parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device used for yolo prediction")

        self.args = parser.parse_args()
        
        print('\n','Input bag file path:', self.args.input)
        print('Input model file path:', self.args.yolo, '\n')

        # test inputs
        if os.path.splitext(self.args.input)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()

        if os.path.splitext(self.args.yolo)[1] != ".pt" and os.path.splitext(self.args.yolo)[1] != ".engine" :
            print("The given file is not of correct file format.")
            print("Only .pt or .engine weights are accepted")
            exit()
        
        self.yolo = YOLO(self.args.yolo)
        self.yolo.fuse()

        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

        # colors for visualization
        
        self._class_to_color = {
            "Person": (0, 0, 255),    
            "Post": (255, 0, 0),
            "Trunk": (0, 255, 0),   
        }
        self.enable_yolo=True
        self.depth_image_units_divisor = 1000
        # pipeline configuration
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            rs.config.enable_device_from_file(self.config, self.args.input, repeat_playback = False) # argument that specifies whether to repeat the playback of the recorded data
            self.device = self.config.resolve(self.pipeline).get_device().as_playback()
            self.device.as_playback().set_real_time( True )
            
            self.config.enable_stream(rs.stream.depth)
            self.config.enable_stream(rs.stream.color)
            
            profile = self.pipeline.start(self.config)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # cv2.namedWindow("Images", cv2.WINDOW_AUTOSIZE)
            self.colorizer = rs.colorizer()

            # Create an align object
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
        finally:
            pass
            
    def predict(self):#, align_depth_msg: Image) -> None:

        while self.enable_yolo: #press ESC in cv2 window to stop
            start_time = time.time()
            frameset = self.pipeline.wait_for_frames()

            aligned_frames = self.align.process(frameset)
            
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()
            color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
            
            results = self.yolo.predict(
                    source=color_image,
                    verbose=False,
                    stream=False,
                    conf=self.args.threshold,
                    device=self.args.device
                )
            results: Results = results[0].cpu()
            
            if results.boxes:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results, aligned_depth_frame)
            
            if results.masks:
                masks = self.parse_masks(results)
            
            if hypothesis is not None or boxes is not None or masks is not None:
                detections = self.parse_detections(results, hypothesis, boxes, masks)
                
                # detections = self.process_detections(detections,aligned_depth_frame)
            end_time = time.time()
            process_time = end_time - start_time
                        
            aligned_depth_image = cv2.cvtColor(np.asanyarray(aligned_depth_frame.get_data()), cv2.COLOR_BGR2RGB)
            depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.06), cv2.COLORMAP_JET)
            
            start_time = time.time()
            
            self.visualize(detections, depth_color_map)#color_image)
            
            visualization_time = time.time() - start_time
            print("Process time: {:.3f} seconds + Visualization time: {:.3f} seconds".format(process_time,visualization_time))
    
    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results, aligned_depth_frame: rs.frame) -> List[BoundingBox2D]:
        
        aligned_depth_np = np.asanyarray(aligned_depth_frame.get_data())
        boxes_list = []

        box_data: Boxes
        for box_data in results.boxes:

            bb = BoundingBox2D()

            # get boxes values
            box = box_data.xyxy[0]
            bb.p1.x = float(box[0]) # p1 left top corner
            bb.p1.y = float(box[1])
            bb.p2.x = float(box[2]) # p2 lower right corner
            bb.p2.y = float(box[3])
            
            roi = aligned_depth_np[ int(bb.p1.y) : int(bb.p2.y), int(bb.p1.x) : int(bb.p2.x)] / \
                    self.depth_image_units_divisor  # convert to meters
            
            # find the z coordinate on the 3D BB
            bb_center_z_coord = aligned_depth_np[int(center_y)][int(
                center_x)] / self.depth_image_units_divisor
            z_diff = np.abs(roi - bb_center_z_coord)
            mask_z = z_diff <= self.maximum_detection_threshold
        
            # if np.any(roi):
            #     continue
            
            print(roi)
            
            # append msg
            boxes_list.append(bb)
            
        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [create_point2d(float(ele[0]), float(ele[1]))
                        for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list
    
    def parse_detections(self, results, hypothesis, boxes, masks):
        
        detections_list = []
        for i in range(len(results)):

            aux_msg = Detection()

            if results.boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]

                aux_msg.bbox = boxes[i]

            if results.masks:
                aux_msg.mask = masks[i]

            detections_list.append(aux_msg)
            
        return detections_list
    
    def process_detections(self, detections_list: List[Detection], aligned_depth_frame ) -> List[Detection]:

        # check if there are detections
        if not detections_list:
            return []

        new_detections = []
        
        aligned_depth_image = cv2.cvtColor(np.asanyarray(aligned_depth_frame.get_data()), cv2.COLOR_BGR2RGB)

        # for detection in detections_list:
        #     bbox3d = self.convert_bb_to_3d(
        #         aligned_depth_image, detection)
            
    def draw_box(self, cv_image: np.array, detection: Detection, color: Tuple[int]) -> np.array:

        # get detection info
        label = detection.class_name
        score = detection.score
        box_msg: BoundingBox2D = detection.bbox
        track_id = detection.id

        min_pt = (round(box_msg.p1.x),
                  round(box_msg.p1.y))
        max_pt = (round(box_msg.p2.x),
                  round(box_msg.p2.y))

        # draw box
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
        
        # write text
        label = "{} ({:.3f})".format(label, score)# str(track_id)
        pos = (min_pt[0] - 5, min_pt[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font,
                    0.6, color, 1, cv2.LINE_AA)

        return cv_image
    
    def draw_mask(self, cv_image: np.array, detection: Detection, color: Tuple[int]) -> np.array:

        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)]
                              for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.6, layer, 0.4, 0, cv_image)
            cv_image = cv2.polylines(cv_image, [mask_array], isClosed=True,
                                     color=color, thickness=1, lineType=cv2.LINE_AA)
        return cv_image
    

    def visualize(self, detections_list: List[Detection], color_image: np.ndarray):
        
        detection: Detection
        for detection in detections_list:

            # random color
            label = detection.class_name
            color = self._class_to_color[label]
            color_image = self.draw_box(color_image, detection, color)
            color_image = self.draw_mask(color_image, detection, color)
            
        cv2.imshow('Image',color_image)
        key = cv2.waitKey(1)
        if key == 27:
            self.enable_yolo = False
        
        
        
        
        # # Colorize depth frame to jet colormap
        # depth_color_frame = colorizer.colorize(aligned_depth_frame)

        # # Convert depth_frame to numpy array to render image in opencv
        # aligned_depth_frame = np.asanyarray(depth_color_frame.get_data())
        

        # overlay = cv2.addWeighted(color_image, 0.5,  aligned_depth_frame, 0.5, 0)
        
        
        # key = cv2.waitKey(1)
        # # if pressed escape exit program
        # 
        # # # Render image in opencv window
        # cv2.imshow("Depth Stream", color_image)
        # key = cv2.waitKey(1)
        # # if pressed escape exit program
        # if key == 27:
        #     cv2.destroyAllWindows()
        #     break

def main():
                
    yolo_model = Yolo()
    yolo_model.predict()

if __name__ == '__main__':
    main()