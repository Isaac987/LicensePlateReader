import os
from PlateDetector import PlateDetector
from PlateCollector import PlateCollector

ROOT = os.path.abspath(".")

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
    "dashcam" : os.path.join(PATHS["data"], "dashcam.mp4"),
}

MODELS = {
    "plates_pt" : os.path.join(PATHS["models"], "plates.pt"),
    "plates_n_pt" : os.path.join(PATHS["models"], "plates_n.pt"),
    "plates_onnx" : os.path.join(PATHS["models"], "plates.onnx"),
}

def main():
    plate_detector = PlateDetector(MODELS["plates_n_pt"])
    plate_collector = PlateCollector(DATA["dashcam"], plate_detector)
    plate_collector()


if (__name__ == "__main__"):
    main()