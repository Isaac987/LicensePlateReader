import numpy as np
import cv2

def ScoreFrame(model, device, frame):
    model.to(device)
    frame = [frame]
    result = model(frame)
    labels, coords = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
    return labels, coords

def ClassToLabel(classes, index):
    return classes[index]

def GetObjectBox(width, height, row):
    x1, y1, x2, y2 = int(row[0] * width), int(row[1] * height), int(row[2] * width), int(row[3] * height)
    return x1, y1, x2, y2

def PlotBoxes(results, frame, classes, thresh):
    labels, coords = results
    n = len(labels)
    width, height = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = coords[i]

        if (row[4] >= thresh):
            x1, y1, x2, y2 = GetObjectBox(width, height, row)
            background = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), background, 2)
        # plate = self.plate_reader.readtext(frame[y1: y2, x1: x2])

    if (len(plate) > 1):
        plate = plate[1]

    cv2.putText(frame, ClassToLabel(classes, labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, background, 2, cv2.LINE_AA)

    return frame