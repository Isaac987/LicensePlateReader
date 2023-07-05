import torch

class PlateDetector:

    def __init__(self, model_name):
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