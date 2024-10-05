import torch.nn as nn
import torch
import torchsummary


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv21 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv22 = nn.Conv2d(channel, channel, 5, padding=2, bias=True)
        self.conv3 = nn.Conv2d(channel * 2, channel, 3, padding=1, bias=True)
        self.sigmoid1 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv21(x)
        x2 = self.conv22(x)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.conv3(x3)
        x3 = self.sigmoid1(x3)
        y = x * x3   # torch.multiply(x, x3)
        y = self.conv4(y)
        return y


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 3, stride=1, padding=3, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class FeaFusAttenBlock(nn.Module):
    def __init__(self, channel):
        super(FeaFusAttenBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, 1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        self.ca1 = CALayer(32)
        self.lka1 = LKA(32)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.ca1(y)
        y = self.lka1(y)
        return y


class SceneAtteBlock(nn.Module):
    def __init__(self, channel):
        super(SceneAtteBlock, self).__init__()
        self.pa = PALayer(channel)
        self.sa = SALayer(channel)

    def forward(self, x):
        y = self.pa(x)
        y = self.sa(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv21 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        x1 = self.conv21(x)
        return y + x1


class HyperDehazeNet(nn.Module):
    def __init__(self):
        super(HyperDehazeNet, self).__init__()
        # 辅助分支
        self.conv21 = nn.Conv2d(41, 32, 3, padding=1, bias=True)
        self.conv22 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        self.ScenAtte1 = SceneAtteBlock(32)

        self.res21 = ResidualBlock(32)
        self.ScenAtte2 = SceneAtteBlock(32)

        self.res22 = ResidualBlock(32)
        self.ScenAtte3 = SceneAtteBlock(32)

        # 主分支
        self.conv1 = nn.Conv2d(305, 32, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)

        self.FusAtte1 = FeaFusAttenBlock(32 + 32)
        self.res1 = ResidualBlock(32)

        self.FusAtte2 = FeaFusAttenBlock(32 + 32)
        self.res2 = ResidualBlock(32)

        self.FusAtte3 = FeaFusAttenBlock(32 + 32)
        self.res3 = ResidualBlock(32)

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 100, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(100, 305, 3, padding=1, bias=True)

    def forward(self, x):
        # 辅助分支
        x_sup = x[:, 246:287, :, :]
        x_sup = self.conv21(x_sup)
        x_sup = self.conv22(x_sup)
        x_ou1 = self.ScenAtte1(x_sup)  # 第一个需要融合的

        x_ou1 = self.res21(x_ou1)
        x_ou2 = self.ScenAtte2(x_ou1)  # 第二个需要融合的

        x_ou2 = self.res22(x_ou2)
        x_ou3 = self.ScenAtte3(x_ou2)  # 第二个需要融合的

        # 主分支
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        y1 = self.FusAtte1(torch.cat([x2, x_ou1], dim=1))
        y1 = self.res1(y1)

        y2 = self.FusAtte2(torch.cat([y1, x_ou2], dim=1))
        y2 = self.res2(y2)

        y3 = self.FusAtte3(torch.cat([y2, x_ou3], dim=1))
        y3 = self.res3(y3)

        y_stack = y1 + y2 + y3
        out = self.conv3(y_stack)
        out = out + x1
        out = self.conv4(out)
        out = self.conv5(out)
        return out


if __name__ == "__main__":
    net = HyperDehazeNet()

    # input_data = torch.rand(1, 305, 512, 512)
    # net = TwobranchNet()
    # out = net(input_data)
    # print(out.shape)

    device = torch.device('cpu')
    net.to(device)

    # torchsummary.summary(net.cuda(), (305, 512, 512))