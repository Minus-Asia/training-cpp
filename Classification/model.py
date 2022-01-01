from torchvision import models
import torch.nn as nn

criterion = nn.CrossEntropyLoss()


def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes, bias=True)
    return model
