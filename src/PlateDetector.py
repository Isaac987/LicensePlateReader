import os
import numpy as np
import cv2
from Plate import Plate

class PlateDetector():
    """Class for detecting license plates in images using a pre-trained object detection model.

    Attributes:
        model_path (str): The path to the pre-trained object detection model in ONNX format.
        input_shape (int): The input shape (both width and height) that the model expects.
        conf_thresh (float, optional): Confidence threshold for filtering out low-confidence predictions. Defaults to 0.7.
        nms_thresh (float, optional): Threshold for non-maximum suppression to remove overlapping bounding boxes with lower confidence scores. Defaults to 0.5.
    """

    def __init__(self, model_path: str, input_shape: int, conf_thresh: float = 0.7, nms_thresh: float = 0.5) -> None:
        """Initialize the PlateDetector object with the specified model and parameters.

        Args:
            model_path (str): The path to the pre-trained object detection model in ONNX format.
            input_shape (int): The input shape (both width and height) that the model expects.
            conf_thresh (float, optional): Confidence threshold for filtering out low-confidence predictions. Defaults to 0.7.
            nms_thresh (float, optional): Threshold for non-maximum suppression to remove overlapping bounding boxes with lower confidence scores. Defaults to 0.5.
        """

        # Check if the model file exists
        if (not os.path.isfile(model_path)):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load the pre-trained ONNX model using OpenCV's dnn module
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)

        # Initialize parameters
        self.model_path: str = model_path
        self.input_shape: int = input_shape
        self.conf_thresh: float = conf_thresh
        self.nms_thresh: float = nms_thresh
        self.scale_w: float = 0.0
        self.scale_h: float = 0.0

    def __repr__(self) -> str:
        """Return a string representation of the PlateDetector object."""

        return f"PlateDetector({self.model_path}, {self.input_shape}, {self.conf_thresh}, {self.nms_thresh})"

    def set_image_size(self, width: float, height: float) -> None:
        """Set the image size used for scaling the bounding box coordinates during prediction.

        Set the image size used for scaling the bounding box coordinates during prediction.
        This should be set before any image prediction or before any video capture.

        Args:
            width (float): The actual width of the image.
            height (float): The actual height of the image.
        """

        self.scale_w = width / self.input_shape
        self.scale_h = height / self.input_shape

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Detect license plates in the input image using the pre-trained object detection model.

        Args:
            img (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: An array of Plate objects representing detected license plates. Each Plate object contains the bounding box coordinates (x1, y1, x2, y2) and the confidence score of the detection.
        """

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
