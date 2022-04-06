from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch


dconv = nn.ConvTranspose2d(in_channels=1, out_channels= 1,  kernel_size=3, stride=1, padding=0,output_padding=0, bias= False)
init.constant(dconv.weight, 1) # 用 1 来填充张量
# print(dconv.weight.shape)


# Variable是torch.autograd中很重要的类。
# 它用来包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息。
# 这里就是使能卷积操作
input = Variable(torch.ones(1, 1, 3, 3))
print(input.shape)
print(dconv(input).shape)