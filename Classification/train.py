import os
import copy
import torch
from tqdm import tqdm
from dataset import Dataset
from model import get_model, criterion

num_epochs = 10
data_dir = "data1"
input_size = 112
num_classes = len(os.listdir(data_dir))
model = get_model(num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
dataset = Dataset(data_dir, input_size, 32)
dataloaders = dataset.dataLoaders
trainer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=1)

for epoch in range(num_epochs):
    save_model = 0
    best_acc_train = 0
    acc_train_cur = 0
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                trainer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        trainer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and (epoch_acc > best_acc or (epoch_acc == best_acc and best_acc_train < acc_train_cur)):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc_train = acc_train_cur
                save_model = epoch
            else:
                acc_train_cur = epoch_acc

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print("best acc", best_acc)

    print('Best val Acc: {:4f}'.format(best_acc))
    print("Best step ", save_model)
    # load best model weights
    model.load_state_dict(best_model_wts)

torch.save(model.state_dict(), "model.pth")
