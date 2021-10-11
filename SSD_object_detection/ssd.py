import torch
from torch import nn
from d2l import torch as d2l


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# We first move this
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)

# see section 13.4.1 d2l book
# sizes is the scale between anchor box width/height and the input image with/height
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
# rations is the aspect ratio of with to height of the anchor box
ratios = [[1, 2, 0.5]] * 5

# totaly we have 4 anchor box on a pixel
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]

        self.blk_0 = base_net()
        self.cls_0 = cls_predictor(idx_to_in_channels[0], num_anchors, num_classes)
        self.bbox_0 = bbox_predictor(idx_to_in_channels[0], num_anchors)

        self.blk_1 = down_sample_blk(64, 128)
        self.cls_1 = cls_predictor(idx_to_in_channels[1], num_anchors, num_classes)
        self.bbox_1 = bbox_predictor(idx_to_in_channels[1], num_anchors)

        self.blk_2 = down_sample_blk(128, 128)
        self.cls_2 = cls_predictor(idx_to_in_channels[2], num_anchors, num_classes)
        self.bbox_2 = bbox_predictor(idx_to_in_channels[2], num_anchors)

        self.blk_3 = down_sample_blk(128, 128)
        self.cls_3 = cls_predictor(idx_to_in_channels[3], num_anchors, num_classes)
        self.bbox_3 = bbox_predictor(idx_to_in_channels[3], num_anchors)

        self.blk_4 = nn.AdaptiveMaxPool2d((1, 1))
        self.cls_4 = cls_predictor(idx_to_in_channels[4], num_anchors, num_classes)
        self.bbox_4 = bbox_predictor(idx_to_in_channels[4], num_anchors)


    def forward(self, X):
        # X : [32, 3, 256, 256]
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        # anchor: [[1, 4096, 4], [1, 1024, 4], [1, 256, 4], [1, 64, 4], [1, 4, 4]]
        # box_preds: [[32, 16, 32, 32], [32, 16, 16, 16], [32, 16, 8, 8], [32, 16, 4, 4], [32, 16, 1, 1]]
        # cls_preds: [[32, 8, 32, 32], [32, 8, 16, 16], [32, 8, 8, 8], [32, 8, 4, 4], [32, 8, 1, 1]]

        Y = self.blk_0(X)
        anchors[0] = d2l.multibox_prior(Y, sizes=sizes[0], ratios=ratios[0])
        cls_preds[0] = self.cls_0(Y)
        bbox_preds[0] = self.bbox_0(Y)

        Y = self.blk_1(Y)
        anchors[1] = d2l.multibox_prior(Y, sizes=sizes[1], ratios=ratios[1])
        cls_preds[1] = self.cls_1(Y)
        bbox_preds[1] = self.bbox_1(Y)

        Y = self.blk_2(Y)
        anchors[2] = d2l.multibox_prior(Y, sizes=sizes[2], ratios=ratios[2])
        cls_preds[2] = self.cls_2(Y)
        bbox_preds[2] = self.bbox_2(Y)

        Y = self.blk_3(Y)
        anchors[3] = d2l.multibox_prior(Y, sizes=sizes[3], ratios=ratios[3])
        cls_preds[3] = self.cls_3(Y)
        bbox_preds[3] = self.bbox_3(Y)

        Y = self.blk_4(Y)
        anchors[4] = d2l.multibox_prior(Y, sizes=sizes[4], ratios=ratios[4])
        cls_preds[4] = self.cls_4(Y)
        bbox_preds[4] = self.bbox_4(Y)

        # anchors: [1, 5444, 4]
        anchors = torch.cat(anchors, dim=1)
        # cls_preds: [32, 10888]
        cls_preds = concat_preds(cls_preds)
        # cls_preds: [32, 5444, 2]
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1,
                                      self.num_classes + 1)
        # bbox_preds: [32, 21776]
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
# This is loss function of the trainer for classification
cls_loss = nn.CrossEntropyLoss(reduction='none')
# This is loss function for the bounding box
bbox_loss = nn.L1Loss(reduction='none')

# This function calculate loss for the target
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    # bbox_labels: [32, 21776], bbox_preds: [32, 21776], bbox_masks: [32, 21776], cls_labels: [32, 5444], cls_preds: [32, 5444, 2],
    # batch_size = 32, num_classes = 2
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]

    #cls_preds: [174208, 2] contain probability that the anchor box being bachground or object
    cls_preds = cls_preds.reshape(-1, num_classes)
    #cls_labels: [174208] the array contains only 0(backgroud) and 1(object)
    cls_labels = cls_labels.reshape(-1)

    # Using Cross Entropy loss to calculate classification loss
    # cls: (32,)
    cls = cls_loss(cls_preds, cls_labels).reshape(batch_size, -1).mean(dim=1)
    # bbox (32,)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

num_epochs = 20


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

