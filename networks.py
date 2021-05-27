import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False):
        super(DoubleConv, self).__init__()
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-5,
                           momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-5,
                           momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_x2(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, n_filters=[32, 64, 128, 256, 512]):
        super(UNet, self).__init__()
        assert len(n_filters) == 5, "length of n_filters should be 5!"
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = DoubleConv(in_channels, n_filters[0])
        self.conv1_0 = DoubleConv(n_filters[0], n_filters[1])
        self.conv2_0 = DoubleConv(n_filters[1], n_filters[2])
        self.conv3_0 = DoubleConv(n_filters[2], n_filters[3])
        self.conv4_0 = DoubleConv(n_filters[3], n_filters[4])

        self.conv3_1 = DoubleConv(n_filters[4], n_filters[3])
        self.conv2_2 = DoubleConv(n_filters[3], n_filters[2])
        self.conv1_3 = DoubleConv(n_filters[2], n_filters[1])
        self.conv0_4 = DoubleConv(n_filters[1], n_filters[0])
        self.up4 = UpConv(n_filters[4], n_filters[3])
        self.up3 = UpConv(n_filters[3], n_filters[2])
        self.up2 = UpConv(n_filters[2], n_filters[1])
        self.up1 = UpConv(n_filters[1], n_filters[0])

        self.conv = nn.Conv2d(n_filters[0], num_classes, 1)

    def forward(self, x, activate=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up3(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up1(x1_3)], dim=1))

        output = self.conv(x0_4)
        return activate(output) if activate else output


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision: bool = False, n_filters: list = [32, 64, 128, 256, 512]):
        super().__init__()
        assert len(n_filters) == 5, "length of n_filters should be 5!"
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = DoubleConv(in_channels, n_filters[0])
        self.conv0_1 = DoubleConv(n_filters[0]+n_filters[1], n_filters[0])
        self.conv0_2 = DoubleConv(n_filters[0]*2+n_filters[1], n_filters[0])
        self.conv0_3 = DoubleConv(n_filters[0]*3+n_filters[1], n_filters[0])
        self.conv0_4 = DoubleConv(n_filters[0]*4+n_filters[1], n_filters[0])

        self.conv1_0 = DoubleConv(n_filters[0], n_filters[1])
        self.conv1_1 = DoubleConv(n_filters[1]+n_filters[2], n_filters[1])
        self.conv1_2 = DoubleConv(n_filters[1]*2+n_filters[2], n_filters[1])
        self.conv1_3 = DoubleConv(n_filters[1]*3+n_filters[2], n_filters[1])

        self.conv2_0 = DoubleConv(n_filters[1], n_filters[2])
        self.conv2_1 = DoubleConv(n_filters[2]+n_filters[3], n_filters[2])
        self.conv2_2 = DoubleConv(n_filters[2]*2+n_filters[3], n_filters[2])

        self.conv3_0 = DoubleConv(n_filters[2], n_filters[3])
        self.conv3_1 = DoubleConv(n_filters[3]+n_filters[4], n_filters[3])

        self.conv4_0 = DoubleConv(n_filters[3], n_filters[4])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(n_filters[0], num_classes, 1)
            self.final2 = nn.Conv2d(n_filters[0], num_classes, 1)
            self.final3 = nn.Conv2d(n_filters[0], num_classes, 1)
            self.final4 = nn.Conv2d(n_filters[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(n_filters[0], num_classes, 1)

    def forward(self, x, activate=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [activate(output1), activate(output2), activate(output3), activate(output4)] if activate else [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return activate(output) if activate else output


class UNetPlusPlus_L1(nn.Module):
    def __init__(self, in_channels, num_classes, n_filters=[32, 64]):
        super().__init__()
        assert len(n_filters) == 2, "length of n_filters should be 2"
        self.conv0_0 = DoubleConv(in_channels, n_filters[0])
        self.conv1_0 = DoubleConv(n_filters[0], n_filters[1])
        self.conv0_1 = DoubleConv(n_filters[1]+n_filters[0], n_filters[0])
        self.final = nn.Conv2d(n_filters[0], num_classes, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, activate=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        output = self.final(x0_1)
        return activate(output) if activate else output


class UNetPlusPlus_L2(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False, n_filters=[32, 64, 128]):
        super().__init__()
        assert len(n_filters) == 3, "length of n_filters should be 2"
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = DoubleConv(in_channels, n_filters[0])
        self.conv1_0 = DoubleConv(n_filters[0], n_filters[1])
        self.conv0_1 = DoubleConv(n_filters[1]+n_filters[0], n_filters[0])

        self.conv2_0 = DoubleConv(n_filters[1], n_filters[2])
        self.conv1_1 = DoubleConv(n_filters[2]+n_filters[1], n_filters[1])
        self.conv0_2 = DoubleConv(n_filters[0]*2+n_filters[1], n_filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(n_filters[0], num_classes, 1)
            self.final2 = nn.Conv2d(n_filters[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(n_filters[0], num_classes, 1)

    def forward(self, x, activate=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output = [activate(output1), activate(output2)
                      ] if activate else [output1, output2]
            return output
        else:
            output = self.final(x0_2)
            return activate(output) if activate else output


class UNetPlusPlus_L3(nn.Module):
    def __init__(self, in_channels, num_classes, deep_supervision=False, n_filters=[32, 64, 128, 256]):
        super().__init__()
        assert len(n_filters) == 4, "length of n_filters should be 2"
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = DoubleConv(in_channels, n_filters[0])
        self.conv1_0 = DoubleConv(n_filters[0], n_filters[1])
        self.conv0_1 = DoubleConv(n_filters[1]+n_filters[0], n_filters[0])

        self.conv2_0 = DoubleConv(n_filters[1], n_filters[2])
        self.conv1_1 = DoubleConv(n_filters[2]+n_filters[1], n_filters[1])
        self.conv0_2 = DoubleConv(n_filters[0]*2+n_filters[1], n_filters[0])

        self.conv3_0 = DoubleConv(n_filters[2], n_filters[3])
        self.conv2_1 = DoubleConv(n_filters[2]+n_filters[3], n_filters[2])
        self.conv1_2 = DoubleConv(n_filters[1]*2+n_filters[2], n_filters[1])
        self.conv0_3 = DoubleConv(n_filters[0]*3+n_filters[1], n_filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(n_filters[0], num_classes, 1)
            self.final2 = nn.Conv2d(n_filters[0], num_classes, 1)
            self.final3 = nn.Conv2d(n_filters[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(n_filters[0], num_classes, 1)

    def forward(self, x, activate=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output = [activate(output1), activate(output2), activate(
                output3)] if activate else [output1, output2, output3]
            return output
        else:
            output = self.final(x0_3)
            return activate(output) if activate else output


def print_param(model,name:str):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("模型:{}\t参数量:{}".format(name, num_params))
