import numpy as np
import cv2
from Plate import Plate

class PlateDetector():

    def __init__(self, model_path: str, input_shape: int, conf_thresh: float = 0.7) -> None:
        self.model_path: str = model_path
        self.input_shape: int = input_shape
        self.conf_thresh = conf_thresh
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)

    def __repr__(self) -> str:
        return f"PlateDetector({self.model_path}, {self.input_shape})"

    def Predict(self, img: np.ndarray) -> np.ndarray:
        img_h: int = img.shape[0]
        img_w: int = img.shape[1]

        # TODO: Move calculations into video capture loops so they are calculated once
        scale_h: float = img_h / self.input_shape
        scale_w: float = img_w / self.input_shape

        blob: np.ndarray = cv2.dnn.blobFromImage(img, 1/255, (self.input_shape, self.input_shape), swapRB=False, crop=False)

        self.model.setInput(blob)
        predictions: np.ndarray = self.model.forward()[0]

        conf_predictions: np.ndarray = predictions[predictions[:, 4] >= self.conf_thresh]

        plates = np.empty(conf_predictions.shape[0], dtype=object)

        for i, prediction in enumerate(conf_predictions):
            confidence: float = prediction[-2]
            x, y, w, h = prediction[:-2]

            factor_w: float = w * 0.5
            factor_h: float =  h * 0.5

            x1: int = int((x - factor_w) * scale_w)
            y1: int = int((y - factor_h) * scale_h)
            x2: int = int((x + factor_w) * scale_w)
            y2: int = int((y + factor_h) * scale_h)

            plates[i] = Plate(x1, y1, x2, y2, confidence)

        return plates
