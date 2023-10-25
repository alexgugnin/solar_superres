import torch
from torch import nn

class DoubleConv(nn.Module):
    '''
    Main bulding block of the net. Consists of two consecutive convolutional
    2D layers with ReLU activation and 25% dropout. It is used in Unet2D as the
    part of paths.
    '''
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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Unet2D(nn.Module):
    '''
    Main architecture of the net. Consists of contracting and expansive paths
    (3 blocks per each, defined by DoubleConv class above), bottom horizontal
    path and skip connections. Constructor method builds a list for every part
    of the structure while forward method performs a propagation.
    '''
    def __init__(self, in_channels:int, out_channels:int, features = [128, 256, 512]):#Orig [16, 32, 64]
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Contracting path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Expansive path
        for feature in reversed(features):
        #Up - transposed might be changed in future
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        #Bottom
        self.bottom = DoubleConv(features[-1], features[-1]*2)
        self.feature_dec_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Forward path protocol. Stands for running all the blocks one by one and
        performing skip connection mechanics.
        '''
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
