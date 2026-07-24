
import torch.nn as nn
import torch
 
 
# ResNet18/34Residual, 23x3Convolution
class BasicBlock(nn.Module):
    # Residual, Convolution, 1
    expansion = 1
 
    # init(): Initialize, Model
    # downsample=NoneResidual, OtherwiseResidual
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # BatchNormalize
        self.bn1 = nn.BatchNorm2d(out_channel)
        # ReLUFunction
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample        
 
    # forward(): ,
    def forward(self, x):
        # ResidualInput
        identity = x
        # IfResidual, Downsample
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # shortcutData
        out += identity
        out = self.relu(out)
        #print(torch.min(out))        
        return out
 
 
# ResNet50/101/152Residual, 1x1+3x3+1x1Convolution
class Bottleneck(nn.Module):
    # expansionResidual, Dimension, 64*4=256
    # BottleneckOutputChannelInput4
    expansion = 4
 
    # init(): Initialize, Model
    # downsample=NoneResidual, OtherwiseResidual, xChannel
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
 
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        # BatchNormalize
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # ReLUFunction
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    # forward(): ,
    def forward(self, x):
        # ResidualInput
        identity = x
        # IfResidual, Downsample
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
        # shortcutData
        out += identity
        out = self.relu(out)
 
        return out
 
'''device = 'cuda'
a = torch.ones(size=(1,1,256,256), device=device, dtype=torch.float)
#model = Decouple_Spatial_net_11_10(inchannels=1, outchannels=3, n_channels=64).to(device)
model = Bottleneck(in_channel=1, out_channel=3).to(device)
b = model(a)
print(b.size())'''


# ResNetClass
class ResNet(nn.Module):
    # InitializeFunction
    def __init__(self,                                              
                 input_dim,                 
                 block=BasicBlock,     
                 blocks_num=[3, 4, 6, 3],
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        
        super(ResNet, self).__init__()
        self.include_top = include_top
        # maxpoolOutputChannel64, ResidualInputChannel64
        self.in_channel = 64
 
        self.groups = groups
        self.width_per_group = width_per_group
 
        self.conv1 = nn.Conv2d(input_dim, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stride=1, stride=2
        # block: Residual
        # block_num: Residual
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # Pooling, Output(H, W), Channel
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # Implementation note.
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Iterate overNetwork
        # nn.ModuleClassMethod:self.modules(), ReturnNetworkmodules
        for m in self.modules():
            # isinstance(object, type): IfClass, isinstance()FunctionReturnTrue
            # IfConvolution
            if isinstance(m, nn.Conv2d):
                # kaimingInitialize, Conv2dConvolutionOutput1
                # fan_in: Weights(Convolution)
                # fan_out: MatrixWeights
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
    # Residual, Residual
    # block: Residual, channel: ConvolutionChannel. block_num: Residual
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # If, Residual
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
 
        layers = []
        #print(self.in_channel, channel)
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        
        self.in_channel = channel * block.expansion        
        #print(self.in_channel, channel)
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # Sequential: Model, Network architecture
        return nn.Sequential(*layers)
 
    # forward(): ,
    def forward(self, x):
        # ResNet, Need
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.maxpool(x)
        #print(x.size())
        # Implementation note.
        x = self.layer1(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.layer3(x)
        #print(x.size())
        x = self.layer4(x)
        #print(x.size())
 
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = nn.Sigmoid()(x)
 
        return x
    
 
if __name__ == "__main__":

    
    # ResNet()blockParameterBasicBlockBottleneck
    # ResNet()blocks_num[0-3][3, 4, 6, 3], ResidualResidual
    # 34resnet
    def resnet34(input_dim, num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet34-333f7ec4.pth
        return ResNet(BasicBlock, [3, 4, 6, 3], input_dim=input_dim, num_classes=num_classes, include_top=include_top)
    
    device = 'cuda'
    a = torch.ones(size=(1,3,256,256), device=device, dtype=torch.float)
    #model = Decouple_Spatial_net_11_10(inchannels=1, outchannels=3, n_channels=64).to(device)
    #model = resnet34(input_dim=3, num_classes=6).to(device)
    model = ResNet(input_dim=3,                 
                 block=BasicBlock,     
                 blocks_num=[3, 4, 6, 3],
                 num_classes=6,
                 include_top=True,
                 groups=1,
                 width_per_group=64).to(device)
    b = model(a)
    print(b.size())


    # 50resnet
    def resnet50(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet50-19c8e357.pth
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    
    
    # 101resnet
    def resnet101(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
