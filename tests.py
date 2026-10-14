import pytest
from yolo import YOLO

def test_outputshape():
    from torch import rand, equal

    input = rand(1, 3, 416, 416) #bs, channels, h, w
    model = YOLO(in_channels=3, num_classes=20)

    print(model(input)[0].shape) #bs, channels, height, width, num_classes+5
    print(model(input)[1].shape)
    print(model(input)[2].shape)

    output0 = (1, 3, 13, 13, 25)
    output1 = (1, 3, 26, 26, 25)
    output2 = (1, 3, 52, 52, 25)

    assert model(input)[0].shape == output0 and model(input)[1].shape == output1 and model(input)[2].shape == output2