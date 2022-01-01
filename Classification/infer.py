import torch
import cv2
import glob
import torch.nn as nn
import time
from PIL import Image
from torchvision import models, transforms
from model import get_model


class Predictor(object):
    def __init__(self, model_path="model.pth", num_classes=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model_ft = get_model(num_classes)
        self.model_ft.load_state_dict(torch.load(model_path))
        self.model_ft.eval()
        self.model_ft.to(self.device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            self.normalize])
        self.classes = ["defect", "good"]

    def label_defective_region(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        inputs = self.transform(img).float()
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs)
        outputs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        outputs = outputs.tolist()
        index = int(preds[0])
        result = {"label": self.classes[index], "score": outputs[0][index]}
        return result


if __name__ == "__main__":
    # img = Image.open(img_path).convert('RGB')
    predict = Predictor(model_path="./label_defective_region.pth")
    lst_img = glob.glob("/mnt/Datasets/YB_2812/bad_defective_region1/*.png")
    num_images = len(lst_img)
    count_incorrect = 0
    for img_path in lst_img:
        img = cv2.imread(img_path)
        start = time.time()
        res = predict.label_defective_region(img)
        print(res)
        if res["label"] == "good" and res["score"] > 0.9:
            count_incorrect += 1
            cv2.imshow("image", img)
            cv2.waitKey(0)

        print(time.time() - start)

    print("Total: ", count_incorrect / num_images)
