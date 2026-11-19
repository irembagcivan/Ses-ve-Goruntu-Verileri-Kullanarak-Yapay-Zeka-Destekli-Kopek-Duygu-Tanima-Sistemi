"""Pixel coordinates; xyxy bounds are exclusive at right and bottom."""
from dataclasses import dataclass
import math


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    if not all(map(math.isfinite, (x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must have finite, positive area")
    return x1, y1, x2-x1, y2-y1


@dataclass(frozen=True)
class CropTransform:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_original(self, xy, resized_size=None):
        width, height = resized_size or (self.x2-self.x1, self.y2-self.y1)
        if width <= 0 or height <= 0:
            raise ValueError("Invalid resized crop dimensions")
        return tuple((self.x1+x*(self.x2-self.x1)/width,
                      self.y1+y*(self.y2-self.y1)/height) for x, y in xy)

    def extract(self, image):
        return image[self.y1:self.y2, self.x1:self.x2].copy()


def padded_crop(box, width, height, margin=0.15):
    x, y, w, h = xyxy_to_xywh(box)
    if width <= 0 or height <= 0 or not math.isfinite(margin) or not 0 <= margin <= 0.5:
        raise ValueError("Positive frame dimensions and margin in [0, 0.5] required")
    left, top = max(0, math.floor(x-w*margin)), max(0, math.floor(y-h*margin))
    right, bottom = min(width, math.ceil(x+w+w*margin)), min(height, math.ceil(y+h+h*margin))
    if right <= left or bottom <= top:
        raise ValueError("Detection does not intersect frame")
    return CropTransform(left, top, right, bottom)
