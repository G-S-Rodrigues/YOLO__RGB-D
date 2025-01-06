import numpy as np
import cv2
import open3d as o3d
import pyglet
import time
import pyrealsense2 as rs
import argparse
import os.path

import threading
import queue

from typing import List, Dict, Tuple

# Vision model
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks

from utils import BoundingBox2D, Point2D, Mask, Detection, BoundingBox3D


class Yolo:

    def __init__(self):

        parser = argparse.ArgumentParser(
            description="Read recorded bag file and display the image with object identification.\
                                            It also obtains the coordinates of the object relative to the camera"
        )

        # Add argument which takes path to a bag file as an input
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            default="/home/guilh/data_tese/vinha-11-07/run1_camera_f.bag",
            help="Path to the bag file",
        )
        parser.add_argument(
            "-y",
            "--yolo_model",
            type=str,
            default="/home/guilh/Robotics4Farmers/Machine_Learning/YOLO_models/r4f_yolov8m_seg.pt",
            help="Path to yolo model",
        )
        parser.add_argument(
            "-pt",
            "--predict_threshold",
            type=float,
            default=0.7,
            help="Threshold for the predictions confidence ",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cuda:0",
            help="Device used for yolo prediction",
        )
        parser.add_argument(
            "-vi",
            "--visualize_image_flag",
            type=bool,
            default=True,
            help="Visualize the image (True/False)",
        )
        parser.add_argument(
            "-vit",
            "--visualize_image_type",
            type=str,
            default="mask",
            help="Visualize the mask or box (mask/box)",
        )
        parser.add_argument(
            "-vpc",
            "--visualize_point_cloud",
            type=bool,
            default=False,
            help="Visualize point cloud (True/False)",
        )
        parser.add_argument(
            "-dt",
            "--display_time",
            type=bool,
            default=False,
            help="Display process time (True/False)",
        )
        parser.add_argument(
            "-sv",
            "--save_video",
            type=bool,
            default=True,
            help="Save the visualization to a mp4 file (True/False)",
        )
        parser.add_argument(
            "-svd",
            "--save_video_dir",
            type=str,
            default="/home/guilh/data_tese/vinha-11-07/videos/run1_camera_f_out.mp4",
            help="Path to the mp4 file",
        )

        self.args = parser.parse_args()

        print("\n", "Input bag file path:", self.args.input)
        print("Input model file path:", self.args.yolo_model, "\n")

        # test inputs
        if os.path.splitext(self.args.input)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()

        if (
            os.path.splitext(self.args.yolo_model)[1] != ".pt"
            and os.path.splitext(self.args.yolo)[1] != ".engine"
        ):
            print("The given file is not of correct file format.")
            print("Only .pt or .engine weights are accepted")
            exit()

        self.yolo = YOLO(self.args.yolo_model)
        self.yolo.fuse()

        # colors for visualization

        self._class_to_color = {
            "Person": (0, 0, 255),
            "Post": (0, 255, 0),
            "Trunk": (255, 0, 0),
        }
        self.enable_yolo = True
        self.depth_to_meters_divisor = 1000
        self.maximum_detection_threshold = 0.3

        # pipeline configuration
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            rs.config.enable_device_from_file(
                self.config, self.args.input, repeat_playback=False
            )  # argument that specifies whether to repeat the playback of the recorded data
            self.device = self.config.resolve(self.pipeline).get_device().as_playback()
            self.device.as_playback().set_real_time(True)

            self.config.enable_stream(rs.stream.depth)
            self.config.enable_stream(rs.stream.color)

            profile = self.pipeline.start(self.config)
            self.camera_intrinsics = (
                profile.get_stream(rs.stream.color)
                .as_video_stream_profile()
                .get_intrinsics()
            )

            self.camera_intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic()
            self.camera_intrinsics_o3d.set_intrinsics(
                self.camera_intrinsics.width,  # msg.width
                self.camera_intrinsics.height,  # msg.height
                self.camera_intrinsics.fx,  # msg.K[0] -> fx
                self.camera_intrinsics.fy,  # msg.K[4] -> fy
                self.camera_intrinsics.ppx,  # msg.K[2] -> cx
                self.camera_intrinsics.ppx,
            )  # msg.K[5] -> cy

            # Create an align object
            align_to = rs.stream.color
            self.align = rs.align(align_to)

        finally:
            pass

    def get_images(self):
        self.start_time_predict = time.time()
        frameset = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frameset)
        aligned_depth_np = np.asanyarray(aligned_frames.get_depth_frame().get_data())
        color_np = np.asanyarray(aligned_frames.get_color_frame().get_data())

        return color_np, aligned_depth_np

    def predict(self, color_np):  # , align_depth_msg: Image) -> None:

        color_image = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

        results = self.yolo.predict(
            source=color_image,
            verbose=False,
            stream=False,
            conf=self.args.predict_threshold,
            device=self.args.device,
        )
        results: Results = results[0].cpu()

        detections = []
        if results.boxes:
            detections = self.parse_hypothesis(results, detections)
            detections = self.parse_boxes(results, detections)

        if results.masks:
            detections = self.parse_masks(results, detections)

        self.process_time = time.time() - self.start_time_predict

        return detections

    def parse_hypothesis(
        self, results: Results, detection_list: List[Detection]
    ) -> List[Detection]:

        box_data: Boxes
        for box_data in results.boxes:
            detection = Detection()
            detection.class_id = int(box_data.cls)
            detection.class_name = self.yolo.names[int(box_data.cls)]
            detection.score = float(box_data.conf)
            detection_list.append(detection)

        return detection_list

    def parse_boxes(
        self, results: Results, detections: List[Detection]
    ) -> List[BoundingBox2D]:

        box_data: Boxes
        for index, box_data in enumerate(results.boxes):

            bb = BoundingBox2D()

            # get boxes values
            box = box_data.xywh[0]
            bb.center.x = float(box[0])
            bb.center.y = float(box[1])
            bb.size.x = float(box[2])
            bb.size.y = float(box[3])

            detections[index].bbox2D = bb

        return detections

    def parse_masks(
        self, results: Results, detections: List[Detection]
    ) -> List[Detection]:

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for index, mask in enumerate(results.masks):

            mask_msg = Mask()

            mask_msg.data = [
                create_point2d(float(point[0]), float(point[1]))
                for point in mask.xy[0].tolist()
            ]

            mask_msg.height = results.orig_img.shape[0]
            mask_msg.width = results.orig_img.shape[1]

            detections[index].mask = mask_msg

        return detections

    def extract_depth(self, detections, aligned_depth_np):
        start_time_extract = time.time()

        if self.args.visualize_image_type == "box":
            detections = self.extract_boxes(detections, aligned_depth_np)

        if self.args.visualize_image_type == "mask":
            detections = self.extract_masks(detections, aligned_depth_np)

        self.extract_time = time.time() - start_time_extract
        return detections

    def extract_one_box(self, bb, aligned_depth_np):
        u_min = max(round(bb.center.x - bb.size.x / 2), 0)
        u_max = min(round(bb.center.x + bb.size.x / 2), aligned_depth_np.shape[1] - 1)
        v_min = max(round(bb.center.y - bb.size.y / 2), 0)
        v_max = min(round(bb.center.y + bb.size.y / 2), aligned_depth_np.shape[0] - 1)

        roi = (
            aligned_depth_np[v_min:v_max, u_min:u_max] / self.depth_to_meters_divisor
        )  # convert to meters

        if not np.any(roi):
            return None

        # find the z coordinate on the 3D BB
        bb_center_z_coord = (
            aligned_depth_np[int(bb.center.y)][int(bb.center.x)]
            / self.depth_to_meters_divisor
        )
        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= self.maximum_detection_threshold

        roi_threshold = roi[mask_z]
        z_min, z_max = np.min(roi_threshold), np.max(roi_threshold)
        z = (z_max + z_min) / 2

        # project from image to world space
        intr = self.camera_intrinsics
        x = z * (bb.center.x - intr.ppx) / intr.fx
        y = z * (bb.center.y - intr.ppy) / intr.fy
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
        return bb_3d

    def extract_boxes(self, detections, aligned_depth_np):

        for detection in detections:
            bb_3d = self.extract_one_box(detection.bbox2D, aligned_depth_np)
            detection.bbox3D = bb_3d

        return detections

    def extract_masks(self, detections, depth_np):
        for detection in detections:
            if detection.class_name == "Person":
                bb_3d = self.extract_one_box(detection.bbox2D, depth_np)
                detection.bbox3D = bb_3d
                continue

            # obtain the pixels correspondents to the mask (mask_idx)
            mask_array_points = np.array(
                [[int(point.x), int(point.y)] for point in detection.mask.data]
            )

            mask_idx = np.zeros_like(depth_np, dtype=np.uint8)
            mask_selection_idx = np.zeros_like(depth_np, dtype=np.uint8)

            if not np.any(mask_array_points):
                detection.mask_box_img = None
                detection.mask_box_world = None
                continue
            cv2.fillPoly(mask_idx, [mask_array_points], 1)

            indices = np.argwhere(mask_idx == 1)
            # select just the lower 1/3 of the mask. The goal is to obtain the base trunk coordinates
            max_index_y, min_index_y = np.max(indices, axis=0), np.min(indices, axis=0)
            mask_height = max_index_y[0] - min_index_y[0]

            # check if the mask id valid ... acceptable size
            if mask_height < 30:  # height in pixels
                detection.mask_box_img = None
                detection.mask_box_world = None
                continue

            mask_selection = round(max_index_y[0] - mask_height / 3)
            mask_selection_idx[mask_selection:, :] = mask_idx[mask_selection:, :]

            # #############################################################3
            # ############     debug visualization ######################
            # cv2.imshow('mask selection debug', (mask_selection_idx * 255).astype(np.uint8))
            # key = cv2.waitKey(0)
            # if key == 27:
            #     break

            # depth values extraction for the mask selection desired
            mask_depth_values = np.array(depth_np[mask_selection_idx == 1])
            mask_depth_values_threshold = (
                self.filter_mask_depth(mask_depth_values) / self.depth_to_meters_divisor
            )
            # mask_depth_values_threshold=mask_depth_values

            if mask_depth_values_threshold.shape[0] == 0:
                detection.mask_box_img = None
                detection.mask_box_world = None
                continue

            indices = np.argwhere(mask_selection_idx == 1)

            x_img = np.median(indices[:, 1]).astype(int)
            y_img = np.median(indices[:, 0]).astype(int)
            z_img = np.median(mask_depth_values_threshold)
            size_x_img = np.max(indices, axis=1) - np.min(indices, axis=1)
            size_y_img = np.max(indices, axis=0) - np.min(indices, axis=0)
            size_z_img = np.max(mask_depth_values_threshold) - np.min(
                mask_depth_values_threshold
            )

            # create 3D BB for the mask in image pixels
            mask_img = BoundingBox3D()
            mask_img.center.x = x_img
            mask_img.center.y = y_img
            mask_img.center.z = z_img
            mask_img.size.x = size_x_img
            mask_img.size.y = size_y_img
            mask_img.size.z = size_z_img
            detection.mask_box_img = mask_img

            # project from image to world space
            intr = self.camera_intrinsics
            x_world = z_img * (x_img - intr.ppx) / intr.fx
            y_world = z_img * (y_img - intr.ppy) / intr.fy
            w_world = z_img * (size_x_img / intr.fx)
            h_world = z_img * (size_y_img / intr.fy)

            # create 3D BB for the mask
            mask_world = BoundingBox3D()
            mask_world.center.x = x_world
            mask_world.center.y = y_world
            mask_world.center.z = z_img
            mask_world.size.x = w_world
            mask_world.size.y = h_world
            mask_world.size.z = size_z_img
            detection.mask_box_world = mask_world

        return detections

    def filter_mask_depth(self, depth_values):

        depth_values = depth_values[depth_values != 0]
        depth_median = np.median(depth_values)

        depth_diff = np.abs((depth_values - depth_median))
        diff_idx = (
            depth_diff
            <= self.maximum_detection_threshold * self.depth_to_meters_divisor
        )

        depth_threshold = depth_values[diff_idx]

        return depth_threshold

    def draw_box(
        self, cv_image: np.array, detection: Detection, color: Tuple[int]
    ) -> np.array:

        # get detection info
        label = detection.class_name
        score = detection.score
        box_msg: BoundingBox2D = detection.bbox2D
        box3d: BoundingBox3D = detection.bbox3D

        min_pt = (
            round(box_msg.center.x - box_msg.size.x / 2),
            round(box_msg.center.y - box_msg.size.y / 2),
        )
        max_pt = (
            round(box_msg.center.x + box_msg.size.x / 2),
            round(box_msg.center.y + box_msg.size.y / 2),
        )

        # draw box
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

        # write text
        label = "{} ({:.3f})".format(label, score)
        pos = (min_pt[0] - 5, min_pt[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font, 0.6, color, 1, cv2.LINE_AA)
        if box3d is not None:
            label = "x: {:.3f}".format(box3d.center.x)
            pos = (max_pt[0] + 2, min_pt[1] + 20)
            cv2.putText(
                cv_image, label, pos, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            label = " y:{:.3f} ".format(box3d.center.y)
            pos = (max_pt[0], min_pt[1] + 40)
            cv2.putText(
                cv_image, label, pos, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            label = " z:{:.3f} ".format(box3d.center.z)
            pos = (max_pt[0], min_pt[1] + 60)
            cv2.putText(
                cv_image, label, pos, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            # label = "size z:{:.3f}".format(box3d.size.z)
            # pos = (max_pt[0] , min_pt[1] + 80)
            # cv2.putText(cv_image,label, pos, font, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.circle(
                cv_image,
                (round(box_msg.center.x), round(box_msg.center.y)),
                5,
                (255, 255, 255),
            )

        return cv_image

    def draw_mask(
        self, cv_image: np.array, detection: Detection, color: Tuple[int]
    ) -> np.array:

        if detection.class_name == "Person":
            self.draw_box(cv_image, detection, color)
            return cv_image

        min_pt = (
            round(detection.bbox2D.center.x - detection.bbox2D.size.x / 2),
            round(detection.bbox2D.center.y - detection.bbox2D.size.y / 2),
        )
        max_pt = (
            round(detection.bbox2D.center.x + detection.bbox2D.size.x / 2),
            round(detection.bbox2D.center.y + detection.bbox2D.size.y / 2),
        )

        label = "{} ({:.3f})".format(detection.class_name, detection.score)
        pos = (min_pt[0] - 5, min_pt[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font, 0.6, color, 1, cv2.LINE_AA)

        mask_msg = detection.mask
        mask_array = np.array([[int(point.x), int(point.y)] for point in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.6, layer, 0.4, 0, cv_image)
            cv_image = cv2.polylines(
                cv_image,
                [mask_array],
                isClosed=True,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        mask_box_img = detection.mask_box_img
        mask_box_world = detection.mask_box_world

        if mask_box_img is not None:
            label = " x: {:.3f} m".format(mask_box_world.center.x)
            pos = (max_pt[0] + 2, min_pt[1] + 20)
            cv2.putText(
                cv_image, label, pos, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            label = " y: {:.3f} m".format(mask_box_world.center.y)
            pos = (max_pt[0], min_pt[1] + 40)
            cv2.putText(
                cv_image, label, pos, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            label = " z: {:.3f} m".format(mask_box_world.center.z)
            pos = (max_pt[0], min_pt[1] + 60)
            cv2.putText(
                cv_image, label, pos, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.circle(
                cv_image,
                (round(mask_box_img.center.x), round(mask_box_img.center.y)),
                5,
                (255, 255, 255),
            )

        return cv_image

    def visualize_image(
        self,
        window_name: str,
        detections_list: List[Detection],
        image: np.ndarray,
        video_writer=None,
    ):

        start_time_visualization = time.time()
        detection: Detection
        img = image
        for detection in detections_list:

            if self.args.visualize_image_type == "box":
                label = detection.class_name
                color = self._class_to_color[label]
                img = self.draw_box(img, detection, color)

            elif self.args.visualize_image_type == "mask":
                label = detection.class_name
                color = self._class_to_color[label]
                img = self.draw_mask(img, detection, color)

        if len(image.shape) == 3:
            color_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow(window_name, color_image)

        elif len(image.shape) == 2:
            depth_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            depth_color_map = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_JET
            )
            cv2.imshow(window_name, depth_color_map)

        # If video_writer is provided, write the depth frame to the video file
        if video_writer is not None:
            video_writer.write(color_image)

        key = cv2.waitKey(1)
        if key == 27:
            self.enable_yolo = False

        self.visualization_time = time.time() - start_time_visualization


def predict_thread(yolo_model, predict_event, predict_queue, image_queue):
    while yolo_model.enable_yolo:
        predict_event.wait()
        predict_event.clear()
        color_np, aligned_depth_np = yolo_model.get_images()
        detections = yolo_model.predict(color_np)
        image_queue.put((color_np, aligned_depth_np))
        predict_queue.put((detections, aligned_depth_np))
        if yolo_model.args.display_time:
            print("Predict time: {:.3f} seconds".format(yolo_model.process_time))


def extract_thread(yolo_model, predict_event, predict_queue, detections_queue):
    while yolo_model.enable_yolo:
        detections, aligned_depth_np = predict_queue.get()
        predict_event.set()
        detections = yolo_model.extract_depth(detections, aligned_depth_np)
        detections_queue.put((detections))
        if yolo_model.args.display_time:
            print(
                "                                Extract time {:.3f} seconds".format(
                    yolo_model.extract_time
                )
            )


def visualization_thread(yolo_model, image_queue, detections_queue):

    if yolo_model.args.save_video:
        frame_width = 1280  # Width of the video frame
        frame_height = 720  # Height of the video frame
        fps = 15  # Frames per second

        print(" Output video dir:", yolo_model.args.save_video_dir)
        print(" width:", frame_width)
        print(" height:", frame_height)
        
        # Initialize the video writer for saving the output
        video_writer = cv2.VideoWriter(
            yolo_model.args.save_video_dir,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        if not video_writer.isOpened():
            print("Error: Video writer failed to open.")
            return

    try:
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
        while yolo_model.enable_yolo:
            color_np, aligned_depth_np = image_queue.get()
            detections = detections_queue.get()
            if yolo_model.args.visualize_image_flag:
                # yolo_model.visualize_image('Depth', detections, aligned_depth_np) # image_name , detections, image
                yolo_model.visualize_image(
                    "Image", detections, color_np, video_writer=video_writer
                )
            if yolo_model.args.display_time:
                print(
                    "                                                                 Visualization time: {:.3f} seconds".format(
                        yolo_model.visualization_time
                    )
                )

    finally:
        # Release the video writer when done
        video_writer.release()
        cv2.destroyAllWindows()


def main():
    predict_event = threading.Event()
    image_queue = queue.Queue(maxsize=10)
    predict_queue = queue.Queue()
    detections_queue = queue.Queue()
    yolo_model = Yolo()
    predict_event.set()

    threads = []
    threads.append(
        threading.Thread(
            target=predict_thread,
            args=(yolo_model, predict_event, predict_queue, image_queue),
        )
    )
    threads.append(
        threading.Thread(
            target=extract_thread,
            args=(yolo_model, predict_event, predict_queue, detections_queue),
        )
    )
    print("Visualize Image:", yolo_model.args.visualize_image_flag)
    if yolo_model.args.visualize_image_flag:
        threads.append(
            threading.Thread(
                target=visualization_thread,
                args=(yolo_model, image_queue, detections_queue),
            )
        )

    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
