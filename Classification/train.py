import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from model import Net
import os
import csv
from skimage import io

import matplotlib.pyplot as plt
import numpy as np


class DatasetCreation(Dataset):
    def __init__(self, data_infos_file, transform=None):
        self.annotations = pd.read_csv(data_infos_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = io.imread(self.annotations.iloc[index, 0])
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, label


train_path = "/media/sf_ShareFolder2/toothbrush/train/"
test_path = "/media/sf_ShareFolder2/toothbrush/test/"

# create training infos for all train data
def create_train_infos(dir_name, filename):
    sub_dirs = []
    for file in os.listdir(dir_name):
        if os.path.isdir(os.path.join(dir_name, file)):
            sub_dirs.append(os.path.join(dir_name, file))

    with open(train_path + filename, 'w') as file:
        writer = csv.writer(file)
        for sub_dir in sub_dirs:
            for image in os.listdir(sub_dir):
                if image.endswith(".png"):
                    img_path = sub_dir + "/" + image
                    data = [img_path, sub_dirs.index(sub_dir)]
                    writer.writerow(data)


create_train_infos(train_path, "train_infos.csv")
create_train_infos(test_path, "test_infos.csv")
# Load training set
batch_size = 4
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize(256),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_set = DatasetCreation(data_infos_file=train_path + "train_infos.csv", transform=transform)
data_set_test = DatasetCreation(data_infos_file=train_path + "test_infos.csv", transform=transform)

trainloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(data_set_test, batch_size=1, shuffle=True, num_workers=4)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# create a CNN model
net = Net()
# define loss function
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
for epoch in range(30):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # print("This is inputs size:", inputs.size())
        # label has size [4]
        # print("This is labels size:", labels.size())
        # Sets the gradients of all optimized torch.Tensors to zero.
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # output: [4, 10], labels: [4]
        # output describe the probability of being each class from 10 classes on cifa-10
        loss = criterion(outputs, labels)
        # calculate derivative of each layer using chain rule
        loss.backward()
        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

print('Finished Training')

# save model
torch.save(net.state_dict(), './model.pth')

############################

# def classify(image_input):
#     output = net(image_input)
#     _, predicted = torch.max(output, 1)
#     return predicted
#
#
# def imshow(img):
#     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# net = Net()
# net.load_state_dict(torch.load("model.pth"))
# dataiter = iter(testloader)
# classes = ('0', '1')
# while True:
#     image, _ = dataiter.next()
#     predicted = classify(image)
#     print(' Predicted: ' + classes[predicted])
#     imshow(torchvision.utils.make_grid(image))
