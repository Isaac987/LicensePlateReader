import time
import numpy as np
import cv2
from Plate import Plate
from PlateDetector import PlateDetector

class PlateCollector:

    def __init__(self, capture_index: str, model: PlateDetector):
        self.capture_index: str = capture_index
        self.model: PlateDetector = model

    def Run(self):
        cap: cv2.VideoCapture = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        prev_time: float = 0.0
        cur_time: float = 0.0
        fps: float = 0.0

        self.model.set_image_size(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while (True):
            ret , frame = cap.read()

            if not ret:
                break

            plates: np.ndarray = self.model.predict(frame)

            cur_time = time.time()
            fps = 1 / (cur_time - prev_time)
            prev_time = cur_time

            cv2.putText(frame, f"fps: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

            for plate in plates:
                text_size = cv2.getTextSize(str(plate), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = int((plate.x1 + plate.x2 - text_size[0]) / 2)
                text_y = int(plate.y1 - 10)

                frame = cv2.rectangle(frame, plate.top_left, plate.bottom_right, (0, 0, 255), 5)
                cv2.putText(frame, str(plate), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Plate Detector", frame)

            if (cv2.waitKey(5) & 0xFF == 27):
                break

        cap.release()
        cv2.destroyAllWindows()