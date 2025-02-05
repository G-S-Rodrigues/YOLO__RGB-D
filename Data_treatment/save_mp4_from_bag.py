import os
import cv2
import numpy as np
import pyrealsense2 as rs  # Intel RealSense SDK

# ========== Configuration ==========
name = "run4_camera_f"
bag_path = f"/media/r4f-orin/SD/Data/vinha-11-07/{name}.bag"
output_video_path = f"/media/r4f-orin/SD/Data/vinha-11-07/videos/{name}.mp4"

# ========== Initialize RealSense Pipeline ==========
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(bag_path, repeat_playback=False)

# Resolve device and check validity
device = cfg.resolve(pipe).get_device()
if not device:
    raise RuntimeError("Error: Cannot open playback device. Verify that the ROSBag file is not corrupted.")

print(f"Processing {device.get_info(rs.camera_info.name)} from {bag_path}\n")

# Start pipeline
profile = pipe.start(cfg)
device.as_playback().set_real_time(False)  # Disable real-time playback to process frames as fast as possible

# Get stream profile to retrieve resolution and FPS
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
video_width, video_height = color_stream.width(), color_stream.height()
video_fps = color_stream.fps()

print(f"Detected Video Resolution: {video_width}x{video_height}, FPS: {video_fps}")

# ========== Initialize Video Writer ==========
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))

if not video_writer.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")

print("Processing frames and writing to video...")

# ========== Process Frames ==========
frame_count = 0
while True:
    success, frameset = pipe.try_wait_for_frames(2000)
    if not success:
        break  # End of video

    color_frame = frameset.get_color_frame()
    if not color_frame:
        continue

    # Convert RealSense frame to OpenCV format
    color_image = np.asanyarray(color_frame.get_data())
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    # Write frame to video
    video_writer.write(color_image_rgb)

    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

print(f"Processing complete! Video saved at {output_video_path}")

# ========== Cleanup ==========
video_writer.release()
pipe.stop()
print("Pipeline stopped and resources released.")
