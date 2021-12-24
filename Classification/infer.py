import torch
import cv2
import glob
import torch.nn as nn
import time
from PIL import Image
from torchvision import models, transforms


class Predictor(object):
    def __init__(self, model_path="model.pth", num_classes=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model_ft = models.resnet18(pretrained=False)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, num_classes)
        self.model_ft.load_state_dict(torch.load(model_path))
        self.model_ft.eval()
        self.model_ft.to(self.device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
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
    predict = Predictor(model_path="label_defective_region.pth")
    for img_path in glob.glob("/home/zsv/PycharmProjects/training-cpp/Classification/data1/val/good/*.png"):
        img = cv2.imread(img_path)
        start = time.time()
        print(predict.label_defective_region(img))
        print(time.time() - start)