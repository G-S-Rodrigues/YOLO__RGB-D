class Point2D():
    def __init__(self):
        self.x = float
        self.y = float

class BoundingBox2D():
    def __init__(self):
        self.p1 = Point2D()
        self.p2 = Point2D()
        
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
        self.bbox = BoundingBox2D()
        # segmentation mask of the detected object
        # it is only the boundary of the segmented object
        mask = Mask
        
