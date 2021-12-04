import torch
import torchvision
from model import Net
import torchvision.transforms as transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

##############################
# Load a test data for example
##############################
# naming of list classes for easier to see
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=True, num_workers=4)

dataiter = iter(testloader)


############################

def classify(image_input):
    output = net(image_input)
    _, predicted = torch.max(output, 1)
    return predicted


def imshow(img):
     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


net = Net()
net.load_state_dict(torch.load("model.pth"))
while True:
    image, _ = dataiter.next()
    predicted = classify(image)
    print(' Predicted: ' + classes[predicted])
    imshow(torchvision.utils.make_grid(image))
