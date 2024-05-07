import numpy as np
import open3d as o3d

# Assuming single_object_depth is your NumPy array containing depth data
# Make sure it's in the correct data type (e.g., uint16)
single_object_depth = np.array([[1000, 2000, 3000],
                                [4000, 5000, 6000],
                                [7000, 8000, 9000]], dtype=np.uint16)

# Convert the NumPy array to an Open3D image object
depth_raw = o3d.geometry.Image(single_object_depth)

# Now you can use depth_raw in your Open3D pipeline
print(depth_raw)
