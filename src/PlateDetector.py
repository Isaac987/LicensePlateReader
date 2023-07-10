import numpy as np
import torch
from Plate import Plate

class PlateDetector:

    def __init__(self, model_name: str) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)

    def Predict(self, image: np.ndarray) -> np.ndarray:
        results: np.ndarray = self.model(image).xyxy[0].numpy()
        print("Results", results)
        plates: np.ndarray = np.empty(results.shape[0], dtype=Plate)
        print("plates", plates)

        for i in np.arange(results.shape[0]):
            plates[i] = Plate(results[i])

        return plates