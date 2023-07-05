from PlateDetector import PlateDetector
import cv2
import easyocr

class PlateCollector:

    def __init__(self, capture_index, plate_detector):
        self.capture_index = capture_index
        self.plate_detector = plate_detector
        self.plate_reader = easyocr.Reader(["en"])

    def plot_boxes(self, results, frame):
        labels, coords = results
        n = len(labels)
        width, height = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = coords[i]

            if (row[4] >= .5):
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
                background = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), background, 2)
                plate = self.plate_reader.readtext(frame[y1: y2, x1: x2])

                if (len(plate) > 1):
                    plate = plate[1]

                cv2.putText(frame, f"{self.plate_detector.class_to_label(labels[i])} {plate}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cv2.LINE_AA)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while (True):
            ret, frame = cap.read()
            assert ret
            # frame = cv2.resize(frame, (420, 420))

            results = self.plate_detector.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            cv2.imshow("Plate Detection", frame)

            if (cv2.waitKey(5) & 0xFF == 27):
                break

        cap.release()