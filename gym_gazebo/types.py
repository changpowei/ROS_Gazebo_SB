from dataclasses import dataclass

@dataclass
class Config:
    viz_image_cv2 : bool
    random_target : bool
    map_name : str
