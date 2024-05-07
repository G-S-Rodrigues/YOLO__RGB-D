import numpy as np
import cv2
import open3d as o3d
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

from utils import BoundingBox2D, Point2D, Mask, Detection, BoundingBox3D


class Yolo():
    
    def __init__(self):
        
        parser = argparse.ArgumentParser(description="Read recorded bag file and display the image with object identification.\
                                            It also obtains the coordinates of the object relative to the camera")

        # Add argument which takes path to a bag file as an input
        parser.add_argument("-i", "--input", type=str, default="/home/guilh/Data_Vineyard_autonomous_robot/Vinha-09-04/row1.bag", help="Path to the bag file")
        parser.add_argument("-y", "--yolo", type=str, default="/home/guilh/Robotics4Farmers/Machine_Learning/YOLO_models/r4f_yolov8m_seg.pt", help="Path to yolo model")
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
        self.maximum_detection_threshold = 0.2
        
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
            self.intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            
            # cv2.namedWindow("Images", cv2.WINDOW_AUTOSIZE)
            self.colorizer = rs.colorizer()

            # Create an align object
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            self.pc = rs.pointcloud()
            
        finally:
            pass
            
    def predict(self):#, align_depth_msg: Image) -> None:

        while self.enable_yolo: #press ESC in cv2 window to stop
            start_time = time.time()
            frameset = self.pipeline.wait_for_frames()
            
            aligned_frames = self.align.process(frameset)
            aligned_depth_np = np.asanyarray(aligned_frames.get_depth_frame().get_data())
            color_np = np.asanyarray(aligned_frames.get_color_frame().get_data())
            
            color_image = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
            
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
                boxes2d, boxes3d = self.parse_boxes(results, aligned_depth_np)
            
            if results.masks:
                masks = self.parse_masks(results)
                a = self.parse_masks_data(results, aligned_depth_np)
                
            if hypothesis is not None or boxes2d is not None or masks is not None:
                detections = self.parse_detections(results, hypothesis, boxes2d, boxes3d, masks)
                
                # detections = self.process_detections(detections,aligned_depth_np)
            end_time = time.time()
            process_time = end_time - start_time
                        
            aligned_depth_image = cv2.cvtColor(aligned_depth_np, cv2.COLOR_BGR2RGB)
            depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.04), cv2.COLORMAP_JET)
            
            start_time = time.time()
            
            depth_raw = o3d.geometry.Image(aligned_depth_np)
            print(depth_raw)
            
            # self.visualize("Depth image",detections, depth_color_map)#color_image)
            # self.visualize("Color_image",detections, color_image)#)
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

    def parse_boxes(self, results: Results, aligned_depth: np.ndarray) -> Tuple[List[BoundingBox2D], List[BoundingBox3D]]:
        
        aligned_depth_np = aligned_depth
        boxes2d_list = []
        boxes3d_list = []

        box_data: Boxes
        for box_data in results.boxes:

            bb = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            bb.center.x = float(box[0])
            bb.center.y = float(box[1])
            bb.size.x = float(box[2])
            bb.size.y = float(box[3])
            
            # crop depth image by the 2d BB
            u_min = max(round(bb.center.x  - bb.size.x / 2), 0)
            u_max = min(round(bb.center.x  + bb.size.x / 2), aligned_depth_np.shape[1] - 1)
            v_min = max(round(bb.center.y  - bb.size.y / 2), 0)
            v_max = min(round(bb.center.y + bb.size.y / 2), aligned_depth_np.shape[0] - 1)

            roi = aligned_depth_np[v_min:v_max, u_min:u_max] / \
                self.depth_image_units_divisor  # convert to meters
                
            if not np.any(roi):
                bb.center.z = 0
                boxes2d_list.append(bb)
                boxes3d_list.append(None)
                continue

            # find the z coordinate on the 3D BB
            bb_center_z_coord = aligned_depth_np[int(bb.center.y)][int(bb.center.x)] / \
                self.depth_image_units_divisor
            z_diff = np.abs(roi - bb_center_z_coord)
            mask_z = z_diff <= self.maximum_detection_threshold
            
            roi_threshold = roi[mask_z]
            z_min, z_max = np.min(roi_threshold), np.max(roi_threshold)
            z = (z_max + z_min) / 2
            
            # project from image to world space
            intr = self.intrinsics
            x = z * (bb.center.x - intr.ppx) / intr.fx
            y = z * ((bb.center.y + bb.size.y / 2) - intr.ppy) / intr.fy
            w = z * (bb.size.x / intr.fx)
            h = z * (bb.size.y / intr.fy)

            # create 3D BB
            bb_3d = BoundingBox3D()
            bb_3d.center.x = x
            bb_3d.center.y = y
            bb_3d.center.z = z
            bb_3d.size.x = w
            bb_3d.size.y = h
            bb_3d.size.z = float(z_max - z_min)
            
            # append msg
            boxes2d_list.append(bb)
            boxes3d_list.append(bb_3d)
            
        return boxes2d_list, boxes3d_list

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
    
    def parse_masks_data(self, results, aligned_depth_np):
        # Get number of objects in the scene
        # object_boxes = results.boxes.xyxy.cpu().numpy()
        # n_objects = object_boxes.shape[0]
        for detection in results:
            # trabalhar aqui
            # resoluç~ao da mask e mask idx nao esta coerente com a imagem 
            np_mask = detection.cpu().numpy().astype(np.uint8)
            print(np_mask)
            
            idx = (np_mask == 1) # The mask is used to create a depth image with only the object
            print(idx.shape)
            mask_depth = np.zeros_like(aligned_depth_np)
            mask_depth[idx] = aligned_depth_np[idx]
            print(mask_depth)
        # for mask in results.masks.data:
            
        #     np_mask = mask.cpu().numpy().astype(np.uint8)
        #     print(np_mask)
            
        #     idx = (np_mask == 1) # The mask is used to create a depth image with only the object
        #     print(idx.shape)
        #     mask_depth = np.zeros_like(aligned_depth_np)
        #     mask_depth[idx] = aligned_depth_np[idx]
        #     print(mask_depth)

                # # Filter pixels which are too far away from the object pixel median?
                # single_object_depth = self.filter_depth_object_img(single_object_depth, idx, 0.15)

                # ### Get the pointcloud for the i'th object
                # depth_raw = o3d.geometry.Image(single_object_depth.astype(np.uint16))
                # object_pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, self.camera_intrinsics)


                # ## Reduce precision of pointcloud to improve performance
                # voxel_grid = object_pointcloud.voxel_down_sample(0.01)
                # voxel_grid, _ = voxel_grid.remove_radius_outlier(nb_points=30, radius=0.05)

                # # Save i'th object pointcloud to list
                # objects_global_point_clouds.append(voxel_grid)


                # ## Extract pointcloud points to numpy array
                # np_pointcloud = np.asarray(object_pointcloud.points)

                # # Get median xyz value
                # median_center = np.median(np_pointcloud, axis=0)
                # median_center = np.append(median_center, 1)
                # median_center_transformed = np.matmul(self.tf_world_to_optical, median_center)

                # # Save i'th object pointcloud median center to list
                # objects_median_center.append(median_center)
                # objects_median_center_transform.append(median_center_transformed)
                
        return 0
    
    
    def parse_detections(self, results, hypothesis, boxes2d, boxes3d, masks):
        
        detections_list = []
        for i in range(len(results)):

            aux_msg = Detection()

            if results.boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]

                aux_msg.bbox2D = boxes2d[i]
                aux_msg.bbox3D = boxes3d[i]
                
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
        box_msg: BoundingBox2D = detection.bbox2D
        box3d: BoundingBox3D = detection.bbox3D
        track_id = detection.id

        min_pt = (round(box_msg.center.x - box_msg.size.x / 2),
                  round(box_msg.center.y - box_msg.size.y / 2))
        max_pt = (round(box_msg.center.x + box_msg.size.x / 2),
                  round(box_msg.center.y + box_msg.size.y / 2))

        # draw box
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)
        
        # write text
        label = "{} ({:.3f})".format(label, score)# str(track_id)
        pos = (min_pt[0] - 5, min_pt[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font, 0.6, color, 1, cv2.LINE_AA)
        if box3d is not None:
            label = "x: {:.3f}".format(box3d.center.x)
            pos = (max_pt[0] + 2 , min_pt[1] + 20)
            cv2.putText(cv_image,label, pos, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            label = " y:{:.3f} ".format(box3d.center.y)
            pos = (max_pt[0] , min_pt[1] + 40)
            cv2.putText(cv_image,label, pos, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            label = " z:{:.3f} ".format(box3d.center.z)
            pos = (max_pt[0] , min_pt[1] + 60)
            cv2.putText(cv_image,label, pos, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            label = "size z:{:.3f}".format(box3d.size.z)
            pos = (max_pt[0] , min_pt[1] + 80)
            cv2.putText(cv_image,label, pos, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(cv_image, (round(box_msg.center.x) , round(box_msg.center.y + box_msg.size.y / 2)), 5, (255,255,255))
            
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
    

    def visualize(self, window_name: str, detections_list: List[Detection], image: np.ndarray):
        
        detection: Detection
        for detection in detections_list:

            # random color
            label = detection.class_name
            color = self._class_to_color[label]
            image = self.draw_box(image, detection, color)
            image = self.draw_mask(image, detection, color)
            
        cv2.imshow(window_name,image)
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