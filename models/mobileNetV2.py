"""
Imported from https://pytorch.org/docs/stable/_modules/torchvision/models/mobilenet.html#mobilenet_v2
and added support for the 1x32x32 mel spectrogram for the speech recognition.

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen: MobileNetV2 - Inverted Residuals and Linear Bottlenecks
https://arxiv.org/abs/1801.04381

Adapted for quantization referencing
https://github.com/Forggtensky/Quantize_Pytorch_Vgg16AndMobileNet/blob/main/MobileNetV2-quantize_all.py
"""

from torch import nn
from torch.quantization import QuantStub, DeQuantStub
import time
import torch


__all__ = ['MobileNetV2', 'mobilenet_v2']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)
        #self.convName = "nn.Sequential(*layers)"
        
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()
        
        self.names = []
        self.times = []

    def hookHelper(self, x, layer):
        if (isinstance(layer, ConvBNReLU)):
            for i in range(0, len(layer)):
                x = self.hookHelper(x, layer[i])
            return x
        else:
            self.names.append(str(layer))
            currentTime = time.time()
            x = (layer)(x)
            self.times.append(time.time() - currentTime)
            return x

    def forward(self, x):
        self.names = []
        self.times = []

        if self.use_res_connect:

            convx = x
            for i in range(0, len(self.conv)):
                convx = self.hookHelper(convx, self.conv[i])
            
            #torch add doesn't work
            x = self.skip_add.add(x, convx)


            return x
        else:
            for i in range(0, len(self.conv)):
                x = self.hookHelper(x, self.conv[i])
            return x


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            in_channels: Input channels for the image (Mel Spectrogram)
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        self.name = "MobileNetV2"
        self.names = []
        self.times = []
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        """features = [ConvBNReLU(3, input_channel, stride=2)]"""
        #self.featuresNames = ["ConvBNReLu(" + str(in_channels) + ", " + str(input_channel) + ", " + str(2) + ")"]
        features = [ConvBNReLU(in_channels, input_channel, stride=2)]

        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                #self.featuresNames.append("Inverted Residual(" + str(input_channel) + ", " + str(output_channel) +
                #                          ", " + str(stride) + ", " + str(t) + ")")
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

                
        # building last several layers"
        #self.featuresNames.append("ConvBNReLu("  + str(in_channels) + ", " + str(self.last_channel) + ", " + str(1) + ")")
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        #self.features.register_forward_hook(getTime)

        # building classifier
        #self.classifierNames = ["Dropout(0.2)", "Linear(" + str(self.last_channel) + ", " + str(num_classes) +")"]
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        #self.classifier.register_forward_hook(getTime)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def hookHelper(self, x, layer):
        if (isinstance(layer, ConvBNReLU)):
            for i in range(0, len(layer)):
                x = self.hookHelper(x, layer[i])
            return x
        elif (isinstance(layer, InvertedResidual)):
            x = layer(x)
            self.names.extend(layer.names)
            self.times.extend(layer.times)
            return x
        else:
            self.names.append(str(layer))
            currentTime = time.time()
            x = (layer)(x)
            self.times.append(time.time() - currentTime)
            return x

    def _forward(self, x):
        self.names = []
        self.times = []

        x = self.quant(x)
        for i in range(0, len(self.features)):
            x = self.hookHelper(x, self.features[i])
                 
        x = x.mean([2, 3])

        for i in range(0, len(self.classifier)):
            x = self.hookHelper(x, self.classifier[i])

        x = self.dequant(x)
        return x


    # Allow for accessing forward method in a inherited class
    forward = _forward
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


def mobilenet_v2(**kwargs):
    """
    Constructs a MobileNetV2 architecture

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    return model
