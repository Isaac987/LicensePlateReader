import os
from PlateDetector import PlateDetector
from PlateCollector import PlateCollector
import cv2
import numpy as np

ROOT = os.path.abspath("..")

PATHS = {
    "data": os.path.join(ROOT, "data"),
    "models": os.path.join(ROOT, "models"),
}

DATA = {
    "img00" : os.path.join(PATHS["data"], "car00.webp"),
    "img01" : os.path.join(PATHS["data"], "car01.webp"),
    "img02" : os.path.join(PATHS["data"], "car02.jpg"),
    "img03" : os.path.join(PATHS["data"], "car03.jpg"),
    "img04" : os.path.join(PATHS["data"], "car04.jpg"),
    "video" : os.path.join(PATHS["data"], "video.mp4"),
}

MODELS = {
    "plates_pt" : os.path.join(PATHS["models"], "plates.pt"),
    "plates_n_pt" : os.path.join(PATHS["models"], "plates_n.pt"),
    "plates_onnx" : os.path.join(PATHS["models"], "plates.onnx"),
    "platesv8_pt" : os.path.join(PATHS["models"], "plates_yolov8.pt"),
    "platesv8_onnx" : os.path.join(PATHS["models"], "plates_yolov8.onnx"),
    "platesv5n_onnx" : os.path.join(PATHS["models"], "plates_yolov5n.onnx"),
}


def main():
    pd = PlateDetector(r"C:\Users\iperkins\Develop\LicensePlateReader\models\plates_yolov5n.onnx", 640)
    pc = PlateCollector(r"C:\Users\iperkins\Develop\LicensePlateReader\data\video.mp4", pd)

    pc.Run()


if (__name__ == "__main__"):
    main()