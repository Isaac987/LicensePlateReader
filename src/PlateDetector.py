import numpy as np
import cv2
from Plate import Plate

class PlateDetector():

    def __init__(self, model_path: str, input_shape: int, conf_thresh: float = 0.7) -> None:
        self.model_path: str = model_path
        self.input_shape: int = input_shape
        self.conf_thresh = conf_thresh
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)
        self.scale_w: float = 0.0
        self.scale_h: float = 0.0

    def __repr__(self) -> str:
        return f"PlateDetector({self.model_path}, {self.input_shape})"

    def SetImageSize(self, width: float, height: float) -> None:
        self.scale_w = width / self.input_shape
        self.scale_h = height / self.input_shape

    def Predict(self, img: np.ndarray) -> np.ndarray:
        blob: np.ndarray = cv2.dnn.blobFromImage(img, 1/255, (self.input_shape, self.input_shape), swapRB=False, crop=False)

        self.model.setInput(blob)
        predictions: np.ndarray = self.model.forward()[0]

        conf_predictions: np.ndarray = predictions[predictions[:, 4] >= self.conf_thresh]

        plates: np.ndarray = np.empty(conf_predictions.shape[0], dtype=object)

        for i, prediction in enumerate(conf_predictions):
            confidence: float = prediction[-2]
            x, y, w, h = prediction[:-2]

            factor_w: float = w * 0.5
            factor_h: float =  h * 0.5

            x1: int = int((x - factor_w) * self.scale_w)
            y1: int = int((y - factor_h) * self.scale_h)
            x2: int = int((x + factor_w) * self.scale_w)
            y2: int = int((y + factor_h) * self.scale_h)

            plates[i] = Plate(x1, y1, x2, y2, confidence)

        return plates
