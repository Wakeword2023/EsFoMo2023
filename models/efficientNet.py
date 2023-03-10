"""
Imported from https://github.com/lukemelas/EfficientNet-PyTorch
and added support for the 1x32x32 mel spectrogram for the speech recognition.

Mingxing Tan, Quoc V. Le: EfficientNet - Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/abs/1905.11946
"""

import torch
from torch import nn
from torch.nn import functional as F

from .utils_efficientNet import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    Swish,
    MemoryEfficientSwish,
)

import time


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        self.names = []
        self.times = []

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._expand_convName = str(self._expand_conv)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._bn0Name = str(self._bn0)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._depthwise_convName = str(self._depthwise_conv)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn1Name = str(self._bn1)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_reduceName = str(self._se_reduce)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
            self._se_expandName = str(self._se_expand)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._project_convName = str(self._project_conv)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn2Name = str(self._bn2)
        self._swish = MemoryEfficientSwish()
        self._swishName = str(self._swish)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        self.names = []
        self.times = []

        # Expansion and Depthwise Convolution
        x = inputs
        print("A round:")
        print("1: " + str(x.size()))
        if self._block_args.expand_ratio != 1:
            self.names.append(self._expand_convName)
            currentTime = time.time()
            x = self._expand_conv(inputs)
            self.times.append(time.time() - currentTime)
            self.names.append(self._bn0Name)
            currentTime = time.time()
            x = self._bn0(x)
            self.times.append(time.time() - currentTime)
            self.names.append(self._swishName)
            currentTime = time.time()
            x = self._swish(x)
            self.times.append(time.time() - currentTime)
            
        print("2: " + str(x.size()))
        print(self._depthwise_conv)
        self.names.append(self._depthwise_convName)
        currentTime = time.time()
        
        x = self._depthwise_conv(x)
        print("3: " + str(x.size()))
        self.times.append(time.time() - currentTime)
        self.names.append(self._bn1Name)
        currentTime = time.time()
        
        print(self._bn1)
        x = self._bn1(x)
        print("3.01: " + str(x.size()))
        self.times.append(time.time() - currentTime)
        self.names.append(self._swishName)
        currentTime = time.time()
        print(self._swish)
        x = self._swish(x)
        self.times.append(time.time() - currentTime)        
        print("3.02: " + str(x.size()))

        # Squeeze and Excitation
        if self.has_se:
            self.names.append("F.adaptive_avg_pool2d(x, 1)")
            currentTime = time.time()
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            print("3.1: " + str(x.size()))
            self.times.append(time.time() - currentTime)
            self.names.append(self._se_reduceName)
            currentTime = time.time()
            x_squeezed = self._se_reduce(x_squeezed)
            print("3.2: " + str(x.size()))
            self.times.append(time.time() - currentTime)
            self.names.append(self._swishName)
            currentTime = time.time()
            x_squeezed = self._swish(x_squeezed)
            print("3.3: " + str(x.size()))
            self.times.append(time.time() - currentTime)
            self.names.append(self._se_expandName)
            currentTime = time.time()
            x_squeezed = self._se_expand(x_squeezed)
            print("3.4: " + str(x.size()))
            self.times.append(time.time() - currentTime)
            self.names.append("torch.sigmoid(x_squeezed)")
            currentTime = time.time()
            x = torch.sigmoid(x_squeezed) * x
            self.times.append(time.time() - currentTime)

        self.names.append(self._project_convName)
        currentTime = time.time()
        x = self._project_conv(x)
        print("4: " + str(x.size()))
        self.times.append(time.time() - currentTime)
        self.names.append(self._bn2Name)
        currentTime = time.time()
        x = self._bn2(x)
        print("4.01: " + str(x.size()))
        self.times.append(time.time() - currentTime)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            print("5: " + str(x.size()))
            print(inputs.size())
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.name = "efficientNet"
        self.names = []
        self.times = []

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters (!)
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 1  # rgb (! originally 3)
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._conv_stemName = str(self._conv_stem)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn0Name = str(self._bn0)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params)) #image size is removed as a parameter
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_headName = str(self._conv_head)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn1Name = str(self._bn1)
        
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._avg_poolingName = str(self._avg_pooling)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._dropoutName = str(self._dropout)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._fcName = str(self._fc)
        self._swish = MemoryEfficientSwish()
        self._swishName = str(self._swish)

        #print(self._blocks)


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    #No extract endpoints!!
            
    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        self.names.append(self._conv_stemName)
        currentTime = time.time()
        x = self._conv_stem(inputs)
        self.times.append(time.time() - currentTime)
        self.names.append(self._bn0Name)
        currentTime = time.time()
        x = self._bn0(x)
        self.times.append(time.time() - currentTime)
        self.names.append(self._swishName)
        currentTime = time.time()
        x = self._swish(x)
        self.times.append(time.time() - currentTime)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            self.names.extend(block.names)
            self.times.extend(block.times)

        # Head
        self.names.append(self._conv_headName)
        currentTime = time.time()
        x = self._conv_head(x)
        self.times.append(time.time() - currentTime)
        self.names.append(self._bn1Name)
        currentTime = time.time()
        x = self._bn1(x)
        self.times.append(time.time() - currentTime)
        self.names.append(self._swishName)
        currentTime = time.time()
        x = self._swish(x)
        self.times.append(time.time() - currentTime)

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        self.names = []
        self.times = []

        
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        self.names.append(self._avg_poolingName)
        currentTime = time.time()
        x = self._avg_pooling(x)
        self.times.append(time.time() - currentTime)
    
        x = x.view(bs, -1) #Replacing if self._global_params.include_top: x = x.flatten(start_dim = 1)

        self.names.append(self._dropoutName)
        currentTime = time.time()
        x = self._dropout(x)
        self.times.append(time.time() - currentTime)

        self.names.append(self._fcName)
        currentTime = time.time()
        x = self._fc(x)
        self.times.append(time.time() - currentTime)
        
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)
    #remove model._change_in_channels(in_channels)

    #Remove from_pretrained

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        valid_models.append('efficientnet-custom')
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
