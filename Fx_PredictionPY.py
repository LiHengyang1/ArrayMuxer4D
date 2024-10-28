import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from convNd import convNd




# ////////////////////////////////////////////////////////////////////////////////////////

class DownResblock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1):
        super(DownResblock, self).__init__()
        self.downsamp = convNd(in_channels, in_channels, 4, 2, 2, 0, use_bias=True)
        self.conv1 = convNd(in_channels, out_channels, 4, 3, 1, 1, use_bias=True,padding_mode='zeros')
        self.conv2 = convNd(out_channels, out_channels, 4, 3, 1, 1, use_bias=True,padding_mode='zeros')
        if use_1x1conv:
            self.conv3 = convNd(in_channels, out_channels, 4, 1, 1, 0, use_bias=True,padding_mode='zeros')
        else:
            self.conv3 = None

    def forward(self, x):
        x = self.downsamp(x)
        Y = F.relu(self.conv1(x))
        Y = self.conv2(Y)
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)


class UpResblock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1, bilinear=False):
        super(UpResblock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = convNd(in_channels, in_channels // 2, num_dims=4, kernel_size=2, stride=2, padding=0,use_bias=True, is_transposed=True)
        self.conv1 = convNd(in_channels, out_channels, 4, 3, 1, 1, use_bias=True, padding_mode='zeros')
        self.conv2 = convNd(out_channels, out_channels, 4, 3, 1, 1, use_bias=True, padding_mode='zeros')
        if use_1x1conv:
            self.conv3 = convNd(in_channels, out_channels, 4, 1, 1, 0, use_bias=True, padding_mode='zeros')
        else:
            self.conv3 = None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        Y = F.relu(self.conv1(x1))
        Y = self.conv2(Y)
        if self.conv3:
            x1 = self.conv3(x1)
        Y += x1
        return F.relu(Y)



class UpResSupp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1, bilinear=True):
        super(UpResSupp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = convNd(in_channels, out_channels, num_dims=4, kernel_size=2, stride=2, padding=0,use_bias=True, is_transposed=True)

        self.conv1 = convNd(out_channels, out_channels, 4, 3, 1, 1, use_bias=True, padding_mode='zeros')
        self.conv2 = convNd(out_channels, out_channels, 4, 3, 1, 1, use_bias=True, padding_mode='zeros')
        if use_1x1conv:
            self.conv3 = convNd(out_channels, out_channels, 4, 1, 1, 0, use_bias=True, padding_mode='zeros')
        else:
            self.conv3 = None

    def forward(self, x):
        x = self.up(x)
        Y = F.relu(self.conv1(x))
        Y = self.conv2(Y)
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)



class ResidualClassic(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=True, strides=1):
        super(ResidualClassic, self).__init__()
        self.conv1 = convNd(in_channels, out_channels, 4, 3, 1, 1, use_bias=True, padding_mode='zeros')
        self.conv2 = convNd(out_channels, out_channels, 4, 3, 1, 1, use_bias=True, padding_mode='zeros')
        if use_1x1conv:
            self.conv3 = convNd(in_channels, out_channels, 4, 1, 1, 0, use_bias=True, padding_mode='zeros')
        else:
            self.conv3 = None


    def forward(self, x):
        Y = F.relu(self.conv1(x))
        Y = self.conv2(Y)
        if self.conv3:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)










class UNet4D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=True):
        super(UNet4D, self).__init__()
        nLayer = [1, 128, 256]
        nLayer0 = [1, 128, 64, 32]

        self.inConv = ResidualClassic(1, nLayer[1], use_1x1conv=True, strides=1)
        self.down1 = DownResblock(nLayer[1], nLayer[2], use_1x1conv=True, strides=1)

        self.up1 = UpResblock(nLayer[2], nLayer[1], use_1x1conv=True, strides=1, bilinear=False)

        self.up01 = UpResSupp(nLayer0[1], nLayer0[2], use_1x1conv=True, strides=1, bilinear=False)
        self.up02 = UpResSupp(nLayer0[2], nLayer0[3], use_1x1conv=True, strides=1, bilinear=False)

        self.outConv = ResidualClassic(nLayer0[3], 1, use_1x1conv=True, strides=1)

    def forward(self, x):
        x1 = self.inConv(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        x = self.up01(x)
        x = self.up02(x)
        x = self.outConv(x)

        return x

#########################################################################################

def pyfun0(image):
    netMAT = UNet4D(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netMAT.to(device)
    netMAT.load_state_dict(torch.load('M4D04.pth', map_location=device))
    netMAT.eval()
    image = image.reshape(1, 1, image.shape[0], image.shape[1], image.shape[2], image.shape[3])
    image = torch.from_numpy(image)
    image = image.to(device=device, dtype=torch.float32)
    tic = datetime.now()
    pred = netMAT(image)
    toc = datetime.now()
    pred = np.array(pred.data.cpu()[0])[0]
    print(np.array((toc - tic).total_seconds()))
    return pred

###################################################################################################################