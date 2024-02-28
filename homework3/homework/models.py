import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


# Define a basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # Downsample if dimensions change
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Define the CNN Classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res_blocks = self._make_layer(64, 64, num_blocks=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64*64, 6)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        super(FCN, self).__init__()
        # Input normalization
        self.normalization = nn.BatchNorm2d(3)

        # Define downsampling layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Define residual blocks
        self.residual_block1 = self._make_residual_block(256, 256)
        self.residual_block2 = self._make_residual_block(256, 256)

        # Define upsampling layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(64, 5, kernel_size=3, padding=1)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        # Apply input normalization
        x = self.normalization(x)

        # Downsample
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))

        # Residual blocks
        x_res = self.residual_block1(x)
        x = x + x_res
        x_res = self.residual_block2(x)
        x = x + x_res

        # Upsample
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))

        # Output convolution
        x = self.conv_out(x)

        return x


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
