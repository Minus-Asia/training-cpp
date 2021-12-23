import torch
import torch.nn as nn
from time import time
from PIL import Image
from torchvision import models, transforms


class infer(object):
    def __init__(self, model_path="model.pth", num_classes=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model_ft = models.resnet18(pretrained=False)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, num_classes)
        self.model_ft.load_state_dict(torch.load(model_path))
        self.model_ft.eval()
        self.model_ft.to(self.device)
        # self.model_ft.cuda()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                         transforms.Resize((112, 112)),
                         transforms.ToTensor(),
                         self.normalize])
        self.classes = ["0", "1"]

    def predict(self, img):
        inputs = self.transform(img).float()
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs)
        print(outputs)
        _, preds = torch.max(outputs, 1)
        return self.classes[int(preds[0])]

if __name__ == "__main__":
    img = Image.open("/home/zsv/PycharmProjects/training-cpp/Classification/data1/val/defect/1640264799.437616.png").convert('RGB')
    predict = infer()
    start = time()
    print(predict.predict(img))
    print(time() - start)
