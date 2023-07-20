import numpy as np
import cv2
from Plate import Plate

class PlateDetector():

    def __init__(self, model_path: str, input_shape: int, conf_thresh: float = 0.7, nms_thresh: float = 0.5) -> None:
        self.model_path: str = model_path
        self.input_shape: int = input_shape
        self.conf_thresh: float = conf_thresh
        self.nms_thresh: float = nms_thresh
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)
        self.scale_w: float = 0.0
        self.scale_h: float = 0.0

    def __repr__(self) -> str:
        return f"PlateDetector({self.model_path}, {self.input_shape})"

    def SetImageSize(self, width: float, height: float) -> None:
        self.scale_w = width / self.input_shape
        self.scale_h = height / self.input_shape

    def Predict(self, img: np.ndarray) -> np.ndarray:

        # Create blob from image and set as model input
        blob: np.ndarray = cv2.dnn.blobFromImage(img, 1/255, (self.input_shape, self.input_shape), swapRB=False, crop=False)
        self.model.setInput(blob)

        # Predict plates from blob
        predictions: np.ndarray = self.model.forward()[0]

        # Keep high probability predictions and convert boxes to xywh format
        predictions = predictions[predictions[:, 4] >= self.conf_thresh]
        predictions[:, 0] -= predictions[:, 2] * 0.5
        predictions[:, 1] -= predictions[:, 3] * 0.5

        # Keep final predictions using non-maximum suppression
        predictions = predictions[cv2.dnn.NMSBoxes(predictions[:, :4], predictions[:, 4], self.conf_thresh, self.nms_thresh)]

        plates: np.ndarray = np.empty(predictions.shape[0], dtype=object)

        for i, prediction in enumerate(predictions):
            confidence: float = prediction[-2]
            x, y, w, h = prediction[:-2]

            # Convert xywh to xyxy format
            x1: int = int(x * self.scale_w)
            y1: int = int(y * self.scale_h)
            x2: int = int((x + w) * self.scale_w)
            y2: int = int((y + h) * self.scale_h)

            plates[i] = Plate(x1, y1, x2, y2, confidence)

        return plates
