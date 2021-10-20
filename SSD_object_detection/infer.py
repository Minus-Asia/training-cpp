import torch
import torchvision
from model import TinySSD
from torch.nn import functional as F
from d2l import torch as d2l


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')


def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    # Predict bounding boxes on classification probability and box prediction using NMS
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]


device = "cuda:0"
net = TinySSD(num_classes=1)
net.load_state_dict(torch.load("model.pt"))
net.to("cuda:0")
net.eval()
img_path = "/mnt/hdd/PycharmProjects/training-cpp/data/banana-detection/bananas_val/images/22.png"
X = torchvision.io.read_image(img_path).unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()

output = predict(X)
print(output)

display(img, output.cpu(), threshold=0.8)
d2l.plt.show()



