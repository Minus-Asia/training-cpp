import json
import os
import cv2


if __name__ == '__main__':
    image_root = "/mnt/Datasets/YB_2812/defect_raw/new"
    json_root = "/mnt/Datasets/YB_2812/defect_json/new"

    defect_dir = "/mnt/Datasets/YB_2812/train/defect"
    good_dir = "/mnt/Datasets/YB_2812/train/good"

    json_lst = sorted(os.listdir(json_root))

    for json_path in json_lst:
        json_path = os.path.join(json_root, json_path)

        # Skip folder
        if os.path.isdir(json_path):
            continue

        print(json_path)

        json_obj = json.load(open(json_path, "r"))

        image_name = os.path.basename(json_obj["imagePath"])
        image_path = os.path.join(image_root, image_name)

        im = cv2.imread(image_path)

        for shape in json_obj["shapes"]:
            label = shape["label"]
            x1, y1 = list(map(int, shape["points"][0]))
            x2, y2 = list(map(int, shape["points"][1]))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            crop = im[y1: y2, x1: x2]

            # cv2.imshow("image", crop)
            # cv2.waitKey(0)

            if label.lower() == "defect":
                current_num = len(os.listdir(defect_dir))
                cv2.imwrite(os.path.join(defect_dir, f"{str(current_num).zfill(3)}.png"), crop)
            else:
                current_num = len(os.listdir(good_dir))
                cv2.imwrite(os.path.join(good_dir, f"{str(current_num).zfill(3)}.png"), crop)
