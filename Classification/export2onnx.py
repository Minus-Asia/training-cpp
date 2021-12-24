import torch
import torch.nn as nn
from torchvision import models


model_ft = models.resnet18(pretrained=False)
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
model_ft.load_state_dict(torch.load("label_defective_region.pth"))
model_ft.eval()

x = torch.randn(1, 3, 112, 112, requires_grad=True)
torch_out = model_ft(x)
torch.onnx.export(model_ft,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "label_defective_region.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                  #               'output': {0: 'batch_size'}}
                  )