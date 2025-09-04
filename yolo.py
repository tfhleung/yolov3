#%%
import torch
import torch.nn as nn

from collections import OrderedDict 

# Basic block used in the YOLO feature pyramid network (FPN).  This class serves as the template and is extended for the ConvBlock and ResidualBlocks.
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Identity()
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.block(x)
        return self.activation(x)

    def _conv_bn(self, in_channels, out_channels, kernel_size, **kwargs):
        return nn.Sequential( OrderedDict( {'conv': nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs),
                                            'bn': nn.BatchNorm2d(num_features=out_channels)}) )

# ConvBlock used in the YOLO FPN.
class ConvBlock(Block):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels)
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

#Darknet53 in the YOLO model used for encoding the image information.
class Darknet53(nn.Module):
    def __init__(self, in_channels, block_size = [64, 128, 256, 512, 1024], num_layers = [1, 2, 8, 8, 4]):
        super().__init__()
        self.num_layers = num_layers

        self.stem = ConvBlock(in_channels, block_size[0] // 2, kernel_size = 3, padding = 1, bias = False)

        self.layers = []
        self.layer_cache = []
        for i, num in enumerate(num_layers):
            for j in range(num + 1): #range is num + 1 b/c we include the first ConvBlock prior to the group of repeated ResidualBlocks
                if j == 0:
                    self.layers.append(ConvBlock(block_size[i] // 2, block_size[i], kernel_size = 3, padding = 1, stride = 2))
                else:
                    self.layers.append(ResidualBlock(block_size[i]))

        self.iter = iter(self.layers)

    def forward(self, x):
        x = self.stem(x)

        self.iter = iter(self.layers)
        for num in self.num_layers:
            for i in range(num + 1):
                x = next(self.iter)(x)
                if i == num - 1:
                    self.layer_cache.append(x)           
        return x

class Detector(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.detector = nn.Sequential(
            ConvBlock(in_channels, 2*in_channels, kernel_size = 3, padding = 1),
            nn.Conv2d(2*in_channels, num_anchors * (num_classes + 5), kernel_size = 1) )

    def forward(self, x):
        x = self.detector(x)
        # return x
        return x.reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
                

#concatenate (in channel direction) with upstream residual layer immediately after upsampling
class Upsample(nn.Module):  #out_channels = in_channels + in_channels // 2
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, in_channels // 2, kernel_size = 1)
        self.upsample = nn.Upsample(scale_factor = 2)

    def forward(self, x, cat_block = None):
        x = self.conv(x)
        x = self.upsample(x)

        if cat_block is not None:
            x = torch.cat([x, cat_block], dim = 1)
        return x

class Scale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size = 1),
            ConvBlock(out_channels, 2*out_channels, kernel_size = 3, padding = 1),
            ResidualBlock(2*out_channels),
            ConvBlock(2*out_channels, out_channels, kernel_size = 1) )
        
    def forward(self, x):
        return self.block(x)

class YOLO(nn.Module):
    def __init__(self, in_channels, num_classes, block_size = [64, 128, 256, 512, 1024], num_layers = [1, 2, 8, 8, 4]):
        super().__init__()
        self.darknet53 = Darknet53(in_channels = in_channels, block_size = block_size, num_layers = num_layers)
        # darknet53 output shape: (1, 1024, 13, 13)

        self.scale1 = Scale(block_size[-1], block_size[-2])

        self.upsampling1 = Upsample(block_size[-2]) #out_channels = in_channels + in_channels // 2
        self.scale2 = Scale(3 * block_size[-3], block_size[-3])

        self.upsampling2 = Upsample(block_size[-3])
        self.scale3 = Scale(3 * block_size[-4], block_size[-4])
        
        self.detector = [Detector(in_channels = block_size[-2], num_classes = num_classes, num_anchors = 3), #input from scale1
                          Detector(in_channels = block_size[-3], num_classes = num_classes, num_anchors = 3), #input from scale2
                          Detector(in_channels = block_size[-4], num_classes = num_classes, num_anchors = 3)] #input from scale3

    def forward(self, x):
        xin = []
        x = self.darknet53(x)

        x = self.scale1(x) #in_channels = 1024, out_channels = 512
        # scale1 output shape: (1, 512, 13, 13)
        xin.append(x)

        x = self.upsampling1(x, cat_block = self.darknet53.layer_cache[-2]) #in_channels = 512, out_channels = 768
        # upsampling1 output shape: (1, 768, 26, 26)
        x = self.scale2(x)
        # scale2 output shape: (1, 256, 26, 26)
        xin.append(x)

        x = self.upsampling2(x, cat_block = self.darknet53.layer_cache[-3])
        # # upsampling2 output shape: (1, 384, 52, 52)
        x = self.scale3(x)
        # # # scale3 output shape: (1, 128, 52, 52)
        xin.append(x)

        output = []
        for i, input in enumerate(xin):
            output.append(self.detector[i](input))

        return output

#%%
    dummy = torch.rand(1, 3, 416, 416)
    model = YOLO(in_channels = 3, num_classes = 20)
    # print(model)
    # print(model(dummy).shape)
    # print(model(dummy))

    print(model(dummy)[0].shape)
    print(model(dummy)[1].shape)
    print(model(dummy)[2].shape)

    # upsample = nn.Upsample(scale_factor=2)
    # print(upsample(model(dummy)).shape)
# torch.Size([2, 3, 13, 13, 25])
# torch.Size([2, 3, 26, 26, 25])
# torch.Size([2, 3, 52, 52, 25])

    # print(f'model.darknet53.layer_cache[-2].shape = {model.darknet53.layer_cache[-2].shape}')
    # print(f'model.darknet53.layer_cache[-3].shape = {model.darknet53.layer_cache[-3].shape}')
    # print(model.darknet53.layer_cache[0].shape)
    # print(model.darknet53.layer_cache[1].shape)
    # print(model.darknet53.layer_cache[2].shape)
    # print(model.darknet53.layer_cache[3].shape)
    # print(model.darknet53.layer_cache[4].shape)

#%%
    dummy1 = torch.rand(1, 512, 26, 26)
    dummy2 = torch.rand(1, 512, 26, 26)

    print(torch.cat([dummy1, dummy2], dim = 1).shape)

    dummy3 = torch.rand(1, 512, 52, 52)
    dummy4 = torch.rand(1, 256, 52, 52)

    print(torch.cat([dummy3, dummy4], dim = 1).shape)


#%%
dummy = torch.rand(1, 3, 416, 416)
print(dummy.shape)

block = Darknet53(in_channels = 3, block_size = [64, 128, 256, 512, 1024], num_layers = [1, 2, 8, 8, 4])
print(block)
# print(block(dummy))
print(block(dummy).shape)
print(len(block.layer_cache))
print(block.layer_cache[0].shape)
print(block.layer_cache[1].shape)
print(block.layer_cache[2].shape)
print(block.layer_cache[3].shape)
print(block.layer_cache[4].shape)

#%%
dummy = torch.rand(1, 3, 416, 416)
print(dummy.shape)
dummy.reshape(3,416,416,1)
dummy.reshape(3,416,1,416)
dummy2 = dummy.reshape(1,416,3,416)
print(dummy2.shape)

#%%
x = torch.ones(1, 3, 512, 512)
# y = 5 * torch.ones(3, 3)
upsample = nn.Upsample(scale_factor=2)
print(upsample(x).shape)

# print(x)
# print(y)
# print(torch.cat([x,y], dim = 0))
# print(torch.cat([x,y], dim = 1))

if __name__ == '__main__':
    pass
# %%
