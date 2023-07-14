import numpy as np
import cv2
from Plate import Plate
from PlateDetector import PlateDetector

class PlateCollector:

    def __init__(self, capture_index: str, model: PlateDetector):
        self.capture_index: str = capture_index
        self.model: PlateDetector = model

    def Run(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while (True):
            ret, frame = cap.read()

            if not ret:
                break

            plates = self.model.Predict(frame)

            for plate in plates:
                frame = cv2.rectangle(frame, plate.top_left, plate.bottom_right, (255, 0, 0), 5)

            cv2.imshow("Plate Detector", frame)

            if (cv2.waitKey(5) & 0xFF == 27):
                break

        cap.release()
        cv2.destroyAllWindows()