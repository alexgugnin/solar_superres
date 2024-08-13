import torch
from torch import nn

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Dropout(p=0.25),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),#, bias=False), -> if Normalisation exists
      nn.ReLU(),
      nn.Dropout(p=0.25)
    )

  def forward(self, x):
    return self.conv(x)

class Unet2D(nn.Module):
  def __init__(self, in_channels, out_channels, features = [128, 256, 512]):
    super().__init__()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    #Down
    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature

    for feature in reversed(features):
    #Up - transposed might be changed in future
      self.ups.append(
        nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
      )
      self.ups.append(DoubleConv(feature*2, feature))

    #bottom
    self.bottom = DoubleConv(features[-1], features[-1]*2)
    self.feature_dec_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

  def forward(self, x):
    skip_connections:list = []

    for block in self.downs:
      x = block(x)
      skip_connections.append(x)
      x = self.pool(x)

    x = self.bottom(x)
    skip_connections = list(reversed(skip_connections))

    for idx in range(0, len(self.ups), 2):
      x = self.ups[idx](x)
      skip_connection = skip_connections[idx//2]
      concat_skip = torch.cat((skip_connection, x), dim=1)
      x = self.ups[idx+1](concat_skip)

    return self.feature_dec_conv(x)