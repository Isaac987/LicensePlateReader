import os
import xml.etree.ElementTree as ET
import cv2
import re


def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    else:
        print(f"Directory '{directory_name}' already exists.")


def extract_filename(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find('.//filename').text
    return filename


def extract_bounding_box(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    x1 = int(root.find('.//xmin').text)
    y1 = int(root.find('.//ymin').text)
    x2 = int(root.find('.//xmax').text)
    y2 = int(root.find('.//ymax').text)

    return x1, y1, x2, y2


def crop_plate(file_in, file_out, x1, y1, x2, y2):
    img = cv2.imread(file_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[y1:y2, x1:x2]
    cv2.imwrite(file_out, img)



def main():
    xml_directory = "annotations"
    car_images_directory = "images"
    plate_images_directory = "cropped_plates"

    create_directory(plate_images_directory)

    xml_files = os.listdir(os.path.join("annotations"))
    xml_files = [xml_files[0]]

    for xml_file in xml_files:
        if (xml_file.endswith(".xml")):
            xml_file_path = os.path.join("annotations", xml_file)
            image_path = extract_filename(xml_file_path)
            match = re.search(r'\d+', image_path)
            img_idx = int(match.group())
            x1, y1, x2, y2 = extract_bounding_box(xml_file_path)
            crop_plate(os.path.join(car_images_directory, image_path), os.path.join(plate_images_directory, f"plate{img_idx}.png"), x1, y1, x2, y2)

if __name__ == "__main__":
    main()