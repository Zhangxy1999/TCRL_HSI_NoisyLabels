import torch
from torch import nn

# import math
# from torchsummary import summary


class SPAM(nn.Module):
    def __init__(self):
        super(SPAM, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(5, 2), stride=(1, 1), padding=(2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, eps=1e-5):
        # feature descriptor on the global spatial information
        N, C, height, width = x.size()
        channel_center = x.view(N, C, -1)[:, :, int((x.shape[2] * x.shape[3] - 1) / 2)]
        channel_center = channel_center.unsqueeze(2)
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_center), dim=2)
        spp = self.conv(t.unsqueeze(1)).transpose(1, 2)
        y = self.sigmoid(spp)

        return x * (1 + y.expand_as(x))


class SPAN(nn.Module):
    def __init__(self, band, classes):
        super(SPAN, self).__init__()
        # dimension reduction
        self.SPAM = SPAM()
        self.conv_DR = nn.Sequential(
                # LKA(band, 30),
                nn.Conv2d(in_channels=band, out_channels=30, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
                nn.ReLU(inplace=True)
            )

        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, padding=(0, 0, 1),
                      kernel_size=(1, 1, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, padding=(0, 0, 1),
                      kernel_size=(1, 1, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, padding=(0, 0, 1),
                      kernel_size=(1, 1, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(32, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv3d_4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=90, padding=(0, 0, 0),
                      kernel_size=(1, 1, 30), stride=(1, 1, 1)),
            nn.BatchNorm3d(90, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.convC = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=120, padding=0, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(120, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(120, classes)
        )

        self.projection = nn.Sequential(
            nn.Linear(120, 120, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(120, 64)
        )

    def forward(self, X):
        X = self.SPAM(X)
        X = self.conv_DR(X)
        spe1 = self.conv3d_1(X.permute(0, 2, 3, 1).unsqueeze(1))
        spe2 = self.conv3d_2(spe1)
        spa1 = self.conv2d_1(X)
        spa2 = self.conv2d_2(spa1)
        spa3 = self.conv2d_3(spa2)
        spe3 = self.conv3d_3(spe2)
        spe4 = self.conv3d_4(spe3)
        spe4 = spe4.squeeze(-1)

        ss = torch.cat((spa3, spe4), dim=1)
        ss = self.convC(ss)
        ss = self.gap(ss)
        ss = ss.view(ss.size(0), -1)

        projection = self.projection(ss)
        cl = self.fc(ss)
        return projection, cl


# model = DwDenseNet(204, 16)
# model.to('cuda')
# summary(model, input_size=(1, 11, 11, 204))

