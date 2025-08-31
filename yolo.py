import torch
import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self, block = ResidualBlock, block_size = [32, 64, 128, 256, 512], num_layers = [2, 4, 16, 16, 8]):
        super().__init__()
        self.stem = []

        self.fpn = nn.Identity()

    def forward(self, x):
        return x

class FPN(nn.Module):
    def __init__(self, block = ResidualBlock, block_size = [32, 64, 128, 256, 512], num_layers = [2, 4, 16, 16, 8]):
        super().__init__()

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding = 0, bias=False)
        self.conv2 = nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding = 1, bias=False)
        self.activation = activation



    def forward(self, x):
        if self.in_channels != self.out_channels:
            identity = self.skip(x)
        else:
            identity = x

        x = self.block(x)
        x += identity
        return self.activation(x)

# Layer Type	Number of Layers	Kernel Size	Stride	Output Size
# Convolutional + BN + Leaky ReLU	1	3x3	1	416 x 416 x 32

# Convolutional + BN + Leaky ReLU	1	3x3	2	208 x 208 x 64
# Residual Block	1 (2 conv layers)	-	-	208 x 208 x 64

# Convolutional + BN + Leaky ReLU	1	3x3	2	104 x 104 x 128
# Residual Block	2 (4 conv layers)	-	-	104 x 104 x 128

# Convolutional + BN + Leaky ReLU	1	3x3	2	52 x 52 x 256
# Residual Block	8 (16 conv layers)	-	-	52 x 52 x 256

# Convolutional + BN + Leaky ReLU	1	3x3	2	26 x 26 x 512
# Residual Block	8 (16 conv layers)	-	-	26 x 26 x 512

# Convolutional + BN + Leaky ReLU	1	3x3	2	13 x 13 x 1024
# Residual Block	4 (8 conv layers)	-	-	13 x 13 x 1024

# Convolutional + BN + Leaky ReLU	

if __name__ == '__main__':
    pass