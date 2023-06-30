import logging
import torch
import cv2

class PlateDetector:

    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        result = self.model(frame)
        labels, coords = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
        return labels, coords
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        labels, coords = results
        n = len(labels)
        width, height = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = coords[i]

            if (row[4] >= .2):
                x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
                background = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), background, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cv2.LINE_AA)
        
        return frame
    
    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while (True):
            ret, frame = cap.read()
            assert ret
            # frame = cv2.resize(frame, (420, 420))

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)

            cv2.imshow("Plate Detection", frame)

            if (cv2.waitKey(5) & 0xFF == 27):
                break
        
        cap.release()

det = PlateDetector(r"C:\Users\iperkins\Develop\LicensePlateReader\data\cars\dashcam.mp4", r"C:\Users\iperkins\Develop\LicensePlateReader\models\plates.pt")
det()