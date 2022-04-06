import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,p_c,i,t,c,n,s):
        super(Block, self).__init__()
        #每个重复的最后一次负责下采样
        #所以i=n-1的时候进行操作
        self.i = i
        self.n = n

        _s = s if i == n-1 else 1#判断是否是最后一次重复的步长，最后一次重复的步长为2
        #判断是否是最后一次重复，最后一次重复负责通道变换为下层的输出
        _c = c if i == n-1 else p_c

        _p_c = p_c*t #输入通道扩增倍数

        self.layer = nn.Sequential(
            nn.Conv2d(p_c,_p_c,1,1,bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c,_p_c,3,_s,padding=1,groups=_p_c,bias=False),
            nn.BatchNorm2d(_p_c),
            nn.ReLU6(),
            nn.Conv2d(_p_c,_c,1,1,bias=False),
            nn.BatchNorm2d(_c)
        )
    def forward(self,x):
        if self.i == self.n-1:
            return self.layer(x)
        else:
            return self.layer(x) + x

class MobileNet_V2(nn.Module):
    def __init__(self):
        super(MobileNet_V2, self).__init__()

        # t c n s
        config52 = [
            [1, 16, 1, 1],
            [3, 64, 2, 2],  # 208 104
            [3, 256, 5, 2],  # 104 52
        ]
        config26 = [
            [3, 512, 5, 2],  # 26 13
        ]
        config13 = [
            [3, 1024, 3, 2]
        ]

        self.input_layer = nn.Sequential(
            nn.Conv2d(3,32,3,2,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        self.blocks52 = []
        p_c = 32
        for t,c,n,s in config52:
            for i in range(n):
                self.blocks52.append(Block(p_c,i,t,c,n,s))
            p_c = c
        self.truck_52 = nn.Sequential(*self.blocks52)

        self.blocks26 = []
        p_c = 256
        for t,c,n,s in config26:
            for i in range(n):
                self.blocks26.append(Block(p_c,i,t,c,n,s))
            p_c = c
        self.truck_26 = nn.Sequential(*self.blocks26)


        self.blocks13 = []
        p_c = 512
        for t,c,n,s in config13:
            for i in range(n):
                self.blocks13.append(Block(p_c,i,t,c,n,s))
            p_c = c
        self.truck_13 = nn.Sequential(*self.blocks13)


    def forward(self,x):
        h52 = self.truck_52(self.input_layer(x))
        h26 = self.truck_26(h52)
        h13 = self.truck_13(h26)

        return h52, h26, h13

if __name__ == '__main__':
    net = MobileNet_V2()
    h52, h26, h13 = net(torch.randn(1,3,416,416))
    print(h52.shape)
    print(h26.shape)
    print(h13.shape)
