import json
import os
import sys
from random import randint
import numpy as np


import cv2

if __name__ == '__main__':
    image_defect_root = "/media/asgard/DATA/Minus_Data/Data/Data_prepare_loikeo/defect"
    image_good_root = "/media/asgard/DATA/Minus_Data/Data/Data_prepare_loikeo/good"
    json_root = "/media/asgard/DATA/Minus_Data/Data/Data_prepare_loikeo/defect_json"

    output_dir = "/media/asgard/DATA/Minus_Data/Data/Data_prepare_loikeo/output"

    json_lst = sorted(os.listdir(json_root))
    image_good_arr = next(os.walk(image_good_root))[2]
    # if len(json_lst) != len(image_good_arr):
    #     print("ERROR: the json files and good images should be the same length")
    #     sys.exit()

    for i in range(len(json_lst)):
        json_path = os.path.join(json_root, json_lst[i])

        # Skip folder
        if os.path.isdir(json_path):
            continue
        print(json_path)
        json_obj = json.load(open(json_path, "r"))

        image_defect_name = os.path.basename(json_obj["imagePath"])
        image_defect_path = os.path.join(image_defect_root, image_defect_name)

        # get a random good image to generate defect
        image_good_path = os.path.join(image_good_root, image_good_arr[randint(0, len(image_good_arr))])

        im_defect = cv2.imread(image_defect_path)
        im_good = cv2.imread(image_good_path)
        black_mask = np.zeros_like(im_good)
        for shape in json_obj["shapes"]:
            label = shape["label"]
            x1, y1 = list(map(int, shape["points"][0]))
            x2, y2 = list(map(int, shape["points"][1]))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            im_good[y1: y2, x1: x2] = im_defect[y1: y2, x1: x2]
            cv2.rectangle(black_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        current_num = len(os.listdir(output_dir))
        cv2.imwrite(os.path.join(output_dir, f"{str(current_num).zfill(4)}.png"), im_good)
        cv2.imwrite(os.path.join(output_dir, f"{str(current_num).zfill(4)}_mask.png"), black_mask)
