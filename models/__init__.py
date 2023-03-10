from .vgg import *
from .resnet import *
from .densenet import *
from .mobileNetV2 import *
from .squeezeNet import *
from .mnasNet import *
from .shuffleNet import *
from .efficientNet import *
from .efficientNet_V2 import *

available_models = [
    'vgg19_bn',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'densenet_bc_100_12', 'densenet_bc_250_24', 'densenet_bc_190_40',
    'mobilenet_v2',
    'squeezenet1_0', 'squeezenet1_1',
    'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
    'custom_shufflenet', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
    'efficientnet_custom', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
    'efficientnet_b6', 'efficientnet_b7',
    'effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl'
]

def create_model(model_name, num_classes, in_channels):
    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet34":
        model = resnet34(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet101":
        model = resnet101(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet152":
        model = resnet152(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_100_12":
        model = DenseNet(depth=100, growthRate=12, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_250_24":
        model = DenseNet(depth=250, growthRate=24, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_190_40":
        model = DenseNet(depth=190, growthRate=40, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "squeezenet1_0":
        model = squeezenet1_0(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "squeezenet1_1":
        model = squeezenet1_1(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "mnasnet0_5":
        model = mnasnet0_5(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "mnasnet0_75":
        model = mnasnet0_75(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "mnasnet1_0":
        model = mnasnet1_0(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "mnasnet1_3":
        model = mnasnet1_3(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "custom_shufflenet":
        model = custom_shufflenet(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "shufflenet_v2_x0_5":
        model = shufflenet_v2_x0_5(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x1_0(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "shufflenet_v2_x1_5":
        model = shufflenet_v2_x1_5(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "shufflenet_v2_x2_0":
        model = shufflenet_v2_x2_0(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "efficientnet_b0":
        model = EfficientNet.from_name('efficientnet-b0')
    elif model_name == "efficientnet_b1":
        model = EfficientNet.from_name('efficientnet-b1')
    elif model_name == "efficientnet_b2":
        model = EfficientNet.from_name('efficientnet-b2')
    elif model_name == "efficientnet_b3":
        model = EfficientNet.from_name('efficientnet-b3')
    elif model_name == "efficientnet_b4":
        model = EfficientNet.from_name('efficientnet-b4')
    elif model_name == "efficientnet_b5":
        model = EfficientNet.from_name('efficientnet-b5')
    elif model_name == "efficientnet_b6":
        model = EfficientNet.from_name('efficientnet-b6')
    elif model_name == "efficientnet_b7":
        model = EfficientNet.from_name('efficientnet-b7')
    elif model_name == "efficientnet_custom":
        model = EfficientNet.from_name('efficientnet-custom')
    elif model_name == "effnetv2_s":
        model = effnetv2_s(num_classes=num_classes)
    elif model_name == "effnetv2_m":
        model = effnetv2_m(num_classes=num_classes)
    elif model_name == "effnetv2_l":
        model = effnetv2_l(num_classes=num_classes)
    elif model_name == "effnetv2_xl":
        model = effnetv2_xl(num_classes=num_classes)
    else:
        model = vgg19_bn(num_classes=num_classes, in_channels=in_channels)
    return model
