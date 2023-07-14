import numpy as np

class Plate:

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float) -> None:
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.confidence: float = confidence
        self.top_left = (self.x1, self.y1)
        self.bottom_right = (self.x2, self.y2)

    def __str__(self) -> str:
        return f"License Plate {100 * self.confidence: .2f}%"

    def __repr__(self) -> str:
        return f"License Plate {self.top_left} {self.bottom_right} {100 * self.confidence: .2f}%"
