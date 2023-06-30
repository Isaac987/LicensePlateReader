import logging
import torch
import cv2

class PlateDetector:

    def __init__(self, capture_index, model_name):
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        return torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)