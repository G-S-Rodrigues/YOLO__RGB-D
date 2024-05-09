class Point2D():
    def __init__(self):
        self.x = float
        self.y = float

class Vector3():
    def __init__(self):
        self.x = float
        self.y = float
        self.z = float

class BoundingBox2D():
    def __init__(self):
        self.center = Point2D()
        self.size = Point2D()

class BoundingBox3D():
    def __init__(self):
        self.center = Vector3()
        self.size = Vector3()
        
class Mask():
    def __init__(self):
        self.height = int
        self.width = int
        self.data = Point2D()
        
class Detection():
    def __init__(self):
        # class probability
        self.class_id = int
        self.class_name = str
        self.score = float
        # ID for tracking
        self.id = str
        # 2D bounding box surrounding the object in pixels
        self.bbox2D = BoundingBox2D()
        self.bbox3D = BoundingBox3D()
        # segmentation mask of the detected object
        # it is only the boundary of the segmented object
        self.mask = Mask()
        self.mask_box_img = BoundingBox3D()
        self.mask_box_world = BoundingBox3D()
