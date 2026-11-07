import numpy as np

# defining a camera class
class camera:
    def __init__(self, location, look_at, focal_length, aperature, near_clip, far_clip, img_side):
        self.location = np.array(location)
        self.look_at = np.array(look_at)
        self.focal_length = focal_length
        self.aperature = aperature
        self.near_clip = near_clip
        self.far_clip = far_clip
        self.img_side = img_side
        
        focal_length_px = (focal_length / aperature) * img_side
        cx, cy = img_side / 2, img_side / 2
        
        self.K = np.array([
            [focal_length_px, 0, cx],
            [0, focal_length_px, cy], 
            [0, 0, 1]
        ])
    
    def __str__(self):
        return f'Camera(location = {self.location}, look_at = {self.look_at})'
    
# class defining a Domelight
class domelight:
    def __init__(self, color, intensity):
        self.color = color
        self.intensity = intensity
    
    def __str__(self):
        return f'DomeLight(color = {self.color}, intensity = {self.intensity})'

# class defining a RectLight
class rectlight:
    def __init__(self, color, intensity, location, width, height):
        self.color = color
        self.intensity = intensity
        self.location = location
        self.width = width
        self.height = height
    
    def __str__(self):
        return f'RectLight(color = {self.color}, intensity = {self.intensity}, location = {self.location}, width = {self.width}, height = {self.height})'
    
# class defining a prim
class usd_prim:
    def __init__(self, location, rotation, scale, type, address):
        self.location = location
        self.rotation = rotation
        self.scale = scale
        self.type = type
        self.address = address
    
    def __str__(self):
        return f'(type = {self.type}, location = {self.location}, rotation = {self.rotation}, scale = {self.scale}, address = {self.address})'

class AddCommand:
    """Represents a single add() command with dynamic attributes"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"AddCommand({attrs})"