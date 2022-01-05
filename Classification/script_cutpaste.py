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
        cropped_img = np.zeros_like(im_good)
        for shape in json_obj["shapes"]:
            if shape["shape_type"] == "rectangle":
                x1, y1 = list(map(int, shape["points"][0]))
                x2, y2 = list(map(int, shape["points"][1]))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                im_good[y1: y2, x1: x2] = im_defect[y1: y2, x1: x2]
                cv2.rectangle(black_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            else:
                black_mask = cv2.cvtColor(black_mask, cv2.COLOR_BGR2GRAY)
                # generate mask
                temp_mask = np.zeros_like(im_good)
                temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_BGR2GRAY)

                shape_points = np.array(shape["points"]).astype(int)
                cv2.fillPoly(black_mask, [shape_points], 255)
                cv2.fillPoly(temp_mask, [shape_points], 255)
                black_mask = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)
                temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR)
                # crop a peace of image
                cropped_img = cv2.subtract(temp_mask, im_defect)
                cropped_img = cv2.subtract(temp_mask, cropped_img)
                temp_mask_inv = cv2.bitwise_not(temp_mask)
                im_good = cv2.bitwise_and(temp_mask_inv, im_good)
                im_good = cv2.bitwise_or(cropped_img, im_good)

        current_num = len(os.listdir(output_dir))
        cv2.imwrite(os.path.join(output_dir, f"{str(current_num).zfill(4)}.png"), im_good)
        cv2.imwrite(os.path.join(output_dir, f"{str(current_num).zfill(4)}_mask.png"), black_mask)

