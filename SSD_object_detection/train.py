from model import TinySSD, calc_loss
from d2l import torch as d2l
import torch

device = "cuda:0"
num_epochs = 30
net = TinySSD(num_classes=1)

batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
net = net.to(device)

for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    # metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        trainer.zero_grad()
        # X: [32, 3, 256, 256], Y: [32, 1, 5]
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        # # anchors: [1, 5444, 4], cls_preds: [32, 5444, 2], bbox_preds: [32, 21776]
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        # bbox_labels: [32, 21776], bbox_masks: [32, 21776], cls_labels: [32, 5444]
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets

        # l: (32,)
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
        print(l.mean().item())
        l.mean().backward()
        trainer.step()

torch.save(net.state_dict(), "model.pt")
