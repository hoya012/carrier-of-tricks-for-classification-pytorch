import os
import torch
import torch.nn as nn
from network.evonorm import EvoNorm
from torchvision.models import resnet

__all__ = [
    'ResNet50'
]

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    def __init__(self, block, layer_config, num_classes=2, norm='batch', zero_init_residual=False):
        super(ResNet, self).__init__()
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d

        self.in_channel = 64

        self.conv = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = norm(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block,  64*block.expansion, layer_config[0], stride=1, norm=norm)
        self.layer2 = self.make_layer(block, 128*block.expansion, layer_config[1], stride=2, norm=norm)
        self.layer3 = self.make_layer(block, 256*block.expansion, layer_config[2], stride=2, norm=norm)
        self.layer4 = self.make_layer(block, 512*block.expansion, layer_config[3], stride=2, norm=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.dense = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)

    def make_layer(self, block, out_channel, num_blocks, stride=1, norm=None):
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d

        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel, stride),
                norm(out_channel),
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample, norm))
        self.in_channel = out_channel
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel, norm=norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.dense(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm=None):
        super(BasicBlock, self).__init__()
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d

        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.norm1 = norm(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.norm2 = norm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm=None):
        super(Bottleneck, self).__init__()
        if norm == 'evonorm':
            norm = EvoNorm
        else:
            norm = nn.BatchNorm2d

        mid_channel= out_channel // self.expansion
        self.conv1 = conv1x1(in_channel, mid_channel)
        self.norm1 = norm(mid_channel)
        self.conv2 = conv3x3(mid_channel, mid_channel, stride)
        self.norm2 = norm(mid_channel)
        self.conv3 = conv1x1(mid_channel, out_channel)
        self.norm3 = norm(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='ResNet50',
                 pretrained=False, pretrained_path=None, norm='batch', zero_init_residual=False):
        super(ResNet50, self).__init__()

        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.H, self.W, self.C = shape

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name + '.pt')

        model = ResNet(Bottleneck, [3,4,6,3], num_classes, norm, zero_init_residual)
        
        if pretrained:
            if pretrained_path is None:
                model = resnet.resnet50(pretrained=True)
                if norm == 'evonorm':
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m = EvoNorm
                        if isinstance(m, nn.ReLU):
                            m = nn.identity
                if zero_init_residual:
                    for m in model.modules():
                        if isinstance(m, resnet.Bottleneck):
                            nn.init.constant_(m.bn3.weight, 0)
            else:
                checkpoint = torch.load(pretrained_path)
                model.load_state_dict(checkpoint)

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.num_features = 512 * Bottleneck.expansion
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(self.num_features, num_classes)
        )

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name + '.pt')
            torch.save(self.state_dict(), checkpoint_path)

    def load(self):
        assert os.path.exists(self.checkpoint_path)
        self.load_state_dict(torch.load(self.checkpoint_path))

    def forward(self, x):
        out = x
        out = self.features(out)
        out = self.classifier(out)
        return out