import torch
import torch.nn as nn
import math
import os

def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(m, "final_bn") and m.final_bn and False
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

class EffHead(nn.Module):
    """EfficientNet head."""

    def __init__(self, w_in, w_out, nc):
        super(EffHead, self).__init__()
        self._construct(w_in, w_out, nc)

    def _construct(self, w_in, w_out, nc):
        # 1x1, BN, Swish
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.conv_swish = Swish()
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout
        if 0.0 > 0.0:
            self.dropout = nn.Dropout(p=0.0)
        # FC
        self.fc = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Swish, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            Swish(),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out):
        super(MBConv, self).__init__()
        self._construct(w_in, exp_r, kernel, stride, se_r, w_out)

    def _construct(self, w_in, exp_r, kernel, stride, se_r, w_out):
        # Expansion ratio is wrt the input width
        self.exp = None
        w_exp = int(w_in * exp_r)
        # Include exp ops only if the exp ratio is different from 1
        if w_exp != w_in:
            # 1x1, BN, Swish
            self.exp = nn.Conv2d(
                w_in, w_exp, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.exp_bn = nn.BatchNorm2d(w_exp, eps=1e-5, momentum=0.1)
            self.exp_swish = Swish()
        # 3x3 dwise, BN, Swish
        self.dwise = nn.Conv2d(
            w_exp,
            w_exp,
            kernel_size=kernel,
            stride=stride,
            groups=w_exp,
            bias=False,
            # Hacky padding to preserve res  (supports only 3x3 and 5x5)
            padding=(1 if kernel == 3 else 2),
        )
        self.dwise_bn = nn.BatchNorm2d(w_exp, eps=1e-5, momentum=0.1)
        self.dwise_swish = Swish()
        # Squeeze-and-Excitation (SE)
        w_se = int(w_in * se_r)
        self.se = SE(w_exp, w_se)
        # 1x1, BN
        self.lin_proj = nn.Conv2d(
            w_exp, w_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lin_proj_bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = (stride == 1) and (w_in == w_out)

    def forward(self, x):
        f_x = x
        # Expansion
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        # Depthwise
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        # SE
        f_x = self.se(f_x)
        # Linear projection
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        # Skip connection
        if self.has_skip:
            # Drop connect
            if self.training and 0.0 > 0.0:
                f_x = nu.drop_connect(f_x, 0.0)
            f_x = x + f_x
        return f_x


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        self._construct(w_in, exp_r, kernel, stride, se_r, w_out, d)

    def _construct(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        # Construct the blocks
        for i in range(d):
            # Stride and input width apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1),
                MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out),
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        # 3x3, BN, Swish
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=1e-5, momentum=0.1)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

""" 
# EfficientNet-B0 Config
STEM_W = 32
STRIDES = [1, 2, 2, 2, 1, 2, 1]
DEPTHS = [1, 2, 2, 3, 3, 4, 1]
WIDTHS = [16, 24, 40, 80, 112, 192, 320]
EXP_RATIOS = [1, 6, 6, 6, 6, 6, 6]
KERNELS = [3, 3, 5, 3, 5, 5, 3]
HEAD_W = 1280
# EfficientNet-B3 Config 
STEM_W = 40
STRIDES = [1, 2, 2, 2, 1, 2, 1]
DEPTHS = [2, 3, 3, 5, 5, 6, 2]
WIDTHS = [24, 32, 48, 96, 136, 232, 384]
EXP_RATIOS = [1, 6, 6, 6, 6, 6, 6]
KERNELS = [3, 3, 5, 3, 5, 5, 3]
HEAD_W = 1536
_C.EN.SE_R = 0.25
_C.EN.DC_RATIO = 0.0
_C.EN.DROPOUT_RATIO = 0.0
"""



class EfficientNet(nn.Module):
    """EfficientNet-B2 model."""

    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='Network',):
        super(EfficientNet, self).__init__()
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.H, self.W, self.C = shape
        
        STEM_W = 32
        STRIDES = [1, 2, 2, 2, 1, 2, 1]
        DEPTHS = [2, 3, 3, 4, 4, 5, 2]
        WIDTHS = [16, 24, 48, 88, 120, 208, 352]
        EXP_RATIOS = [1, 6, 6, 6, 6, 6, 6]
        KERNELS = [3, 3, 5, 3, 5, 5, 3]
        HEAD_W = 1408

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')

        self._construct(
            stem_w=STEM_W,
            ds=DEPTHS,
            ws=WIDTHS,
            exp_rs=EXP_RATIOS,
            se_r=0.25,
            ss=STRIDES,
            ks=KERNELS,
            head_w=HEAD_W,
            nc=1000,
        )
        self.apply(init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        # Group params by stage
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        # Construct the stem
        self.stem = StemIN(3, stem_w)
        prev_w = stem_w
        # Construct the stages
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            self.add_module(
                "s{}".format(i + 1), EffStage(prev_w, exp_r, kernel, stride, se_r, w, d)
            )
            prev_w = w
        # Construct the head
        self.prev_w = prev_w
        self.head_w = head_w
        self.head = EffHead(prev_w, head_w, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    def save(self, checkpoint_name=''):
        if checkpoint_name == '':
            torch.save(self.state_dict(), self.checkpoint_path)
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name + '.pt')
            torch.save(self.state_dict(), checkpoint_path)

    def load(self):
        assert os.path.exists(self.checkpoint_path)
        self.load_state_dict(torch.load(self.checkpoint_path))