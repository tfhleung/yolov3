#%%
import torch
import torch.nn as nn

from collections import OrderedDict 

# Basic block used in the YOLO feature pyramid network (FPN).  This class serves as the template and is extended for the ConvBlock and ResidualBlocks.
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation = nn.LeakyReLU(0.1), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.block = nn.Identity()
        self.activation = activation
    
    def forward(self, x):
        x = self.block(x)
        return self.activation(x)

    def _conv_bn(self, in_channels, out_channels, kernel_size, **kwargs):
        return nn.Sequential( OrderedDict( {'conv': nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
                                            'bn': nn.BatchNorm2d(num_features=out_channels)}) )

# ConvBlock used in the YOLO FPN.
class ConvBlock(Block):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size)
        self.block = self._conv_bn(in_channels, out_channels, kernel_size, **kwargs)

# ResidualBlock used in the YOLO FPN and consists of two convolutional layers in series.  The number of features maps is halved in the first
# conv-layer and doubled in the subsequent layer so that the number of feature maps at the input/output of the ResidualBlock is identical.
class ResidualBlock(Block):
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels, in_channels // 2, **kwargs)
        self.block = nn.Sequential(
            self._conv_bn(in_channels, in_channels // 2, kernel_size = 1, bias = False, **kwargs),
            self.activation,
            self._conv_bn(in_channels // 2, in_channels, kernel_size = 3, padding = 1, bias = False, **kwargs))

    def forward(self, x):
        identity = x
        x = self.block(x)
        x += identity
        return self.activation(x)

#FPN  in the YOLO model used for encoding the image information.
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, block = ResidualBlock, block_size = [64, 128, 256, 512, 1024], num_layers = [2, 4, 16, 16, 8]):
        super().__init__()
        self.stem = ConvBlock(in_channels, block_size[0] // 2, kernel_size = 3, padding = 1, bias = False)

        self.list = []
        for i, num in enumerate(num_layers):
            for j in range(num + 1):
                # print(j)
                if j == 0:
                    self.list.append(ConvBlock(block_size[i] // 2, block_size[i], kernel_size = 3, padding = 1, stride = 2))
                else:
                    self.list.append(ResidualBlock(block_size[i], stride = 1))

        self.layers = nn.Sequential(*self.list)

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return x

class Detector(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.detector = nn.Sequential(
            ConvBlock(in_channels, 2*in_channels, kernel_size = 3, padding = 1, stride = 1),
            nn.Conv2d(2*in_channels, num_anchors * (num_classes + 5), kernel_size = 1) )
            # ConvBlock(2*in_channels, num_anchors * (num_classes + 5), kernel_size = 1) )

    def forward(self, x):
        x = self.detector(x)
        return x

#concatenate (in channel direction) with upstream residual layer immediately after upsampling
class Upsample(nn.Module):
    def __init__(self, scale_factor = 2, **kwargs):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor, **kwargs)

class YOLO(nn.Module):
    def __init__(self, in_channels, num_classes, block_size = [64, 128, 256, 512, 1024], num_layers = [1, 2, 8, 8, 4]):
        super().__init__()
        self.fpn = FeaturePyramidNetwork(in_channels = in_channels, block_size = block_size, num_layers = num_layers)

        self.post = nn.Sequential(
            ConvBlock(block_size[-1], block_size[-1] // 2, kernel_size = 1, stride = 1, bias = False),
            ConvBlock(block_size[-1] // 2, block_size[-1], kernel_size = 3, padding = 1, stride = 1, bias = False) )

        self.scale1 = nn.Sequential(
            ResidualBlock(block_size[-1], stride = 1, bias = False),
            ConvBlock(block_size[-1], block_size[-1] // 2, kernel_size = 1, stride = 1) )
        
        self.upsampling1 = nn.Sequential(
            ConvBlock(block_size[-1] // 2, block_size[-1] // 4, kernel_size = 1, stride = 1),
            Upsample(scale_factor=2),
            ConvBlock(block_size[-1] // 2 + block_size[-1] // 4, block_size[-1] // 4, kernel_size = 1, stride = 1),
            ConvBlock(block_size[-1] // 4, block_size[-1] // 2, kernel_size = 3, padding = 1, stride = 1)       
        )

        self.scale2 = nn.Sequential(
            ResidualBlock(block_size[-1] // 2, stride = 1, bias = False),
            ConvBlock(block_size[-1] // 2, block_size[-1] // 4, kernel_size = 1, stride = 1) )

        self.upsampling2 = nn.Sequential(
            ConvBlock(block_size[-1] // 4, block_size[-1] // 8, kernel_size = 1, stride = 1),
            Upsample(scale_factor=2),
            ConvBlock(block_size[-1] // 4 + block_size[-1] // 8, block_size[-1] // 8, kernel_size = 1, stride = 1),
            ConvBlock(block_size[-1] // 8, block_size[-1] // 4, kernel_size = 3, padding = 1, stride = 1)       
        )

        self.scale3 = nn.Sequential(
            ResidualBlock(block_size[-1] // 4, stride = 1, bias = False),
            ConvBlock(block_size[-1] // 4, block_size[-1] // 8, kernel_size = 1, stride = 1) )

        #follows from scale1
        self.dectector1 = Detector(in_channels = block_size[-1] // 2, num_classes = num_classes, num_anchors = 3)
        
        #follows from scale2
        self.dectector2 = Detector(in_channels = block_size[-1] // 4, num_classes = num_classes, num_anchors = 3)

        #follows from scale3
        self.dectector3 = Detector(in_channels = block_size[-1] // 8, num_classes = num_classes, num_anchors = 3)


        self.detector = {'small': Detector(block_size[-1]),
                         'mid': Detector(),
                         'large': Detector()}

    def forward(self, x):
        x = self.fpn(x)
        return x


#%%
dummy = torch.rand(2, 3, 416, 416)
print(dummy.shape)

block = ResidualBlock(3, stride = 1)
print(block(dummy).shape)

#%%
dummy = torch.rand(1, 3, 416, 416)
print(dummy.shape)

block = FeaturePyramidNetwork(in_channels = 3, block_size = [64, 128, 256, 512, 1024], num_layers = [1, 2, 8, 8, 4])
print(block)
# print(block(dummy))
print(block(dummy).shape)

#%%
dummy = torch.rand(1, 3, 416, 416)
print(dummy.shape)
dummy.reshape(3,416,416,1)
dummy.reshape(3,416,1,416)
dummy2 = dummy.reshape(1,416,3,416)
print(dummy2.shape)

#%%
x = torch.ones(1, 3, 3)
# y = 5 * torch.ones(3, 3)
upsample = nn.Upsample(scale_factor=2, dim = 0)
print(upsample(x))

# print(x)
# print(y)
# print(torch.cat([x,y], dim = 0))
# print(torch.cat([x,y], dim = 1))

if __name__ == '__main__':
    pass
# %%
