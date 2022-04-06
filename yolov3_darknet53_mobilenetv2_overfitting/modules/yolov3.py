import torch

from modules.darknet53 import *
from modules.mobilenetv2 import MobileNet_V2

class Yolov3(torch.nn.Module):

    def __init__(self, use_mobilenetv2:bool):
        super(Yolov3, self).__init__()
        self.use_mobilenetv2 = use_mobilenetv2

        if self.use_mobilenetv2 == True:
            self.trunk = MobileNet_V2()
        else:
            self.trunk = Darknet53()

        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.detetion_13 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 24, 1, 1, 0)
        )

        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 3, 1, 1),
            UpsampleLayer()
        )

        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(768, 256)
        )

        self.detetion_26 = torch.nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 24, 1, 1, 0)
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 3, 1, 1),
            UpsampleLayer()
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(384, 128)
        )

        self.detetion_52 = torch.nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 24, 1, 1, 0)
        )

    def forward(self, x):

        h_52, h_26, h_13 = self.trunk(x)

        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)

        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)

        return detetion_out_13, detetion_out_26, detetion_out_52

if __name__ == '__main__':
    yolo = Yolov3(use_mobilenetv2=False)
    x = torch.randn(1,3,416,416)

    y = yolo(x)

    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
