import cv2
import numpy
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# img = cv2.imread(r"../VOC2007/JPEGImages/000017.jpg")/255  # 读取图片
# # img = cv2.resize(img, (416, 416))  # 对图片进行变形
# batch_img = img.transpose(2, 0, 1)[None, ...]  # hwc 换成 1chw  ??????????????????????????????????
# print(batch_img.shape)
# print(batch_img)


print("------------------------- Image trans归一化-------------------------------")
img = Image.open(r"../VOC2007/JPEGImages/000017.jpg")
# img = img.resize((416, 416))
trans = transforms.ToTensor()
batch_img = trans(img)[None, ...]
print(batch_img.shape)
print(batch_img)
print("------------------------- Image 手动归一化-------------------------------")


img = numpy.array(img)
img = img.transpose((2, 0, 1))[None,...]
print(img.shape)
print(img/255)


print("------------------------- cv2 手动归一化-------------------------------")
img = cv2.imread(r"../VOC2007/JPEGImages/000017.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)
img = img.transpose((2, 0, 1))[None,...]
print(img.shape)
print(img/255)
exit()


# #####################################################################################
# import torchvision.transforms as transforms
# import numpy as np
# import torch
#
# # 定义转换方式，transforms.Compose将多个转换函数组合起来使用
#
#
# # 定义一个数组
# d1 = [1,2,3,4,5,6]
# d2 = [4,5,6,7,8,9]
# d3 = [7,8,9,10,11,14]
# d4 = [11,12,13,14,15,15]
# d5 = [d1,d2,d3,d4]
# d = np.array([d5,d5,d5],dtype=np.float32)
#
#
# transform1 = transforms.Compose([transforms.ToTensor()])  #归一化到(0,1)，简单直接除以255
# d_t = np.transpose(d,(1,2,0)) # 转置为类似图像的shape，(H,W,C)，作为transform的输入
# print('d.shape: ',d.shape, '\n', 'd_t.shape: ', d_t.shape)
# d_t_trans = transform1(d_t) # 直接使用函数归一化
#
# # 手动归一化,下面的两个步骤可以在源码里面找到
# d_t_temp = torch.from_numpy(d_t.transpose((2,0,1)))
# d_t_trans_man = d_t_temp.float().div(255)
#
# print(d_t_trans.equal(d_t_trans_man))
#
#
# # transforms.Normalize()作用
# # 在transforms.Compose([transforms.ToTensor()])中加入transforms.Normalize()，如下所示：
# # transforms.Compose([transforms.ToTensor(),transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))])，
# # 则其作用就是先将输入归一化到(0,1)，再使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
#
# # transform2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))])# 归一化到（0,1）之后，再 (x-mean)/std，归一化到（-1,1），数据中存在大于mean和小于mean
# #
# # d_t_trans_2 = transform2(d_t)
# # d_t_temp1 = torch.from_numpy(d_t.transpose((2,0,1)))
# # d_t_temp2 = d_t_temp1.float().div(255)
# # d_t_trans_man2 = d_t_temp2.sub_(0.5).div_(0.5)
# # print(d_t_trans_2.equal(d_t_trans_man2))
