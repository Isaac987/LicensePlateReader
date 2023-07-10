import numpy as np

class Plate:

    def __init__(self, xyxy_result: np.ndarray) -> None:
        coords = np.ndarray = xyxy_result[:-2].astype(int)

        self.x1: int = coords[0]
        self.y1: int = coords[1]
        self.x2: int = coords[2]
        self.y2: int = coords[3]
        self.confidence: float = xyxy_result[-2]
        self.top_left = (self.x1, self.y1)
        self.bottom_right = (self.x2, self.y2)

    def __str__(self) -> str:
        return f"License Plate {100 * self.confidence: .2}%"
    
    def __repr__(self) -> str:
        return f"License Plate ({self.top_left}) ({self.bottom_right}) {100 * self.confidence: .2}%"
        