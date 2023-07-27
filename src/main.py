import os
import argparse
from PlateDetector import PlateDetector
from PlateCollector import PlateCollector

ROOT = os.path.abspath(".")
PLATE_MODEL = os.path.join(ROOT, "models", "yolov5_nano.onnx")

parser = argparse.ArgumentParser(prog="PlateSpotter", description="License plate detection tool. Press ESC to exit the program.")
parser.add_argument("-i", "--input", default=0, metavar="input_source", help="The input path of a video or 0 for camera capture (default: 0)")


def main():
    pd = PlateDetector(PLATE_MODEL, 640)
    pc = PlateCollector(parser.parse_args().input, "PlateSpotter", pd)

    pc.Run()


if (__name__ == "__main__"):
    main()