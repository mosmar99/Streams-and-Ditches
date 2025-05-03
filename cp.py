import torch
from src.models import unet
from torchviz import make_dot

in_channels = 1
classes = [0, 1, 2]


dummy_input = torch.randn(1, in_channels, 500, 500)

model = unet.UNet(in_channels=1, classes=classes)
model.load('./checkpoints/tf2_minloss_checkpoint.pth.tar')

dummy_input = torch.randn(1, 1, 500, 500)
output = model(dummy_input)

dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("model_visualization", format="png")