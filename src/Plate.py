import numpy as np

class Plate:
    """Class representing a detected license plate.

    Attributes:
        x1 (int): x-coordinate of the top-left corner of the license plate bounding box.
        y1 (int): y-coordinate of the top-left corner of the license plate bounding box.
        x2 (int): x-coordinate of the bottom-right corner of the license plate bounding box.
        y2 (int): y-coordinate of the bottom-right corner of the license plate bounding box.
        confidence (float): Confidence score of the detection.
        top_left (Tuple[int, int]): Tuple representing the (x, y) coordinates of the top-left corner of the bounding box.
        bottom_right (Tuple[int, int]): Tuple representing the (x, y) coordinates of the bottom-right corner of the bounding box.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float) -> None:
        """Initialize the Plate object.

        Args:
            x1 (int): x-coordinate of the top-left corner of the license plate bounding box.
            y1 (int): y-coordinate of the top-left corner of the license plate bounding box.
            x2 (int): x-coordinate of the bottom-right corner of the license plate bounding box.
            y2 (int): y-coordinate of the bottom-right corner of the license plate bounding box.
            confidence (float): Confidence score of the detection.
        """

        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x2
        self.y2: int = y2
        self.confidence: float = confidence
        self.top_left = (self.x1, self.y1)
        self.bottom_right = (self.x2, self.y2)

    def __str__(self) -> str:
        """Return a string representation of the Plate object.

        Returns:
            str: A string with information about the detected license plate.
        """

        return f"License Plate {100 * self.confidence: .2f}%"

    def __repr__(self) -> str:
        """Return a string representation of the Plate object.

        Returns:
            str: A string with detailed information about the detected license plate, including bounding box coordinates and confidence.
        """

        return f"Plate({self.x1}, {self.y1}, {self.x2}, {self.y2}, {self.confidence: .2f})"
