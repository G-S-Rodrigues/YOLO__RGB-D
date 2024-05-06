import os
import cv2
import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import ffmpeg

import matplotlib.pyplot as plt  # 2D plotting library producing publication quality figures
from IPython.display import clear_output  # Clear the screen
# extract from bag nice script (example)
# https://github.com/IntelRealSense/librealsense/issues/4934#issuecomment-537705225



name = "swincarTripVinha"
bag_path = "/home/guilh/Data_Vineyard_autonomous_robot/Vinha_swincar/" + name + ".bag"
output_video_path = "/home/guilh/Data_Vineyard_autonomous_robot/Vinha_swincar/videos/" + name + ".mp4"
video_fps=60


############## Creating a Pipeline ############################
# The pipeline is a high-level API for streaming and processing frames, abstracting camera configurations
# and simplifying user interaction with the device and computer vision processing modules.
# Config is a utility object used by a pipeline.

pipe = rs.pipeline()  # Create a pipeline
cfg = rs.config()  # Create a default configuration
print("\nPipeline is created\n")


############## Find RealSense Devices ############################
print("Creating devices from records..\n")
#cfg.enable_device_from_file(str(os.environ['HOME'])+'//downloads//bear.bag', repeat_playback = True)
cfg.enable_device_from_file(bag_path, repeat_playback = False)

device = cfg.resolve(pipe).get_device()
if not device:
    print("Cannot open playback device. Verify that the ROSBag input file is not corrupted")
else:
    print(device.get_info(rs.camera_info.name)+'\n')
    
############## Find Depth and RGB Sensors ############################
rgb_sensor = depth_sensor = None
                        
print("Available sensors:")
for s in device.sensors:                             
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        print(" - found RGB sensor")
        rgb_sensor = s                                # Set RGB sensor
    if s.get_info(rs.camera_info.name) == 'Stereo Module':
        depth_sensor = s                              # Set Depth sensor
        print(" - found Depth sensor")
    
############## Displaying Depth and Color Frames ############################

profile = pipe.start(cfg)                                 # Configure and start the pipeline
device.as_playback().set_real_time( False )               # Set to False to read each frame sequentially
print("Bag is being converted to jpg:")

counter = 0
success = True
while success:                                            # Increase to display more frames
    success, frameset = pipe.try_wait_for_frames(1000)    # Read frames from the file, packaged as a frameset
    if success == False:
        print("Images are converted!")
        break
    color_frame = frameset.get_color_frame()              # Get RGB frame
    color = np.asanyarray(color_frame.get_data())
    plt.imsave("images/img_%0*d.jpg" %(5, counter), np.array(color))
    counter += 1
    
pipe.stop()                                               # Stop the pipeline
print("Pipeline stopped and video is being created!")

# creates the video
print("Creating video")

(
ffmpeg
.input('images/*.jpg', pattern_type='glob', framerate=video_fps)
.output(output_video_path)
.run()
)

# delete images

for filename in os.listdir('images'):
    if os.path.isfile(os.path.join('images', filename)):
        os.remove(os.path.join('images', filename))

print("images deleted")