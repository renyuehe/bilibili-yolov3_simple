import os
import time

from torch import nn

import dataset
from modules.yolov3 import *
from tool.utils import make_image_data

def loss_fn(output, target, alpha):
    output = output.permute(0, 2, 3, 1)#N,45,13,13==>N,13,13,45
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)#N,13,13,3,15
    # print("output:",output.shape)
    mask_obj = target[..., 0] > 0#N,13,13,3
    # print("mask_obj:",mask_obj.shape)
    mask_noobj = target[..., 0] == 0
    # print("mask_noobj:",mask_noobj.shape)
    # print("output[mask_obj]:",output[mask_obj].shape)
    # print("output[mask_noobj]:", output[mask_noobj].shape)

    loss_p_fun=nn.BCELoss()
    loss_p=loss_p_fun(torch.sigmoid(output[...,0]),target[...,0])

    loss_box_fun=nn.MSELoss()
    loss_box=loss_box_fun(output[mask_obj][...,1:5],target[mask_obj][...,1:5])

    loss_cls_fun=nn.CrossEntropyLoss()
    loss_cls=loss_cls_fun(output[mask_obj][...,5:],torch.argmax(target[mask_obj][...,5:],dim=1,keepdim=True).squeeze(dim=1))

    loss = alpha * loss_p + (1-alpha)*0.6*loss_box+ (1-alpha)*0.4*loss_cls
    print("loss_p:{}, loss_box:{}, loss_cls:{}".format(loss_p, loss_box, loss_cls))
    return loss


# from tool import focalloss
# def loss_fn(output, target, alpha):
#     output = output.permute(0, 2, 3, 1)#N,45,13,13==>N,13,13,45
#     output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)#N,13,13,3,15
#
#     # 正样本 和 部分样本
#     mask_obj = target[..., 0] > 0#N,13,13,3
#     # 负样本
#     mask_noobj = target[..., 0] == 0
#
#     # 二分类 focal_loss
#     loss_p_fun=focalloss.BCEFocalLoss()
#     loss_p=loss_p_fun(torch.sigmoid(output[...,0]),target[...,0])
#
#     loss_box_fun=nn.MSELoss()
#     loss_box=loss_box_fun(output[mask_obj][...,1:5],target[mask_obj][...,1:5])
#
#     # 多分类 focal_loss
#     loss_cls_fun=focalloss.MultiCEFocalLoss(class_num=3)
#     loss_cls=loss_cls_fun(output[mask_obj][...,5:],torch.argmax(target[mask_obj][...,5:],dim=1,keepdim=True).squeeze(dim=1))
#
#     loss = loss_p + loss_box + loss_cls
#     print("loss_p:{}, loss_box:{}, loss_cls:{}".format(loss_p, loss_box, loss_cls))
#     return loss

from common import *
import time
if __name__ == '__main__':
    time_start = time.time()
    myDataset = dataset.MyDataset()
    train_loader = torch.utils.data.DataLoader(myDataset, batch_size=2, shuffle=True)

    net = Yolov3(use_mobilenetv2=is_use_mobilenetv2).cuda()
    if os.path.exists(yolov3_net_pt_name):
        net.load_state_dict(torch.load(yolov3_net_pt_name))
        print("加载成功")

    net.train()
    opt = torch.optim.Adam(net.parameters())

    times = 0
    while True:
        for target_13, target_26, target_52, img_data in train_loader:
            target_13, target_26, target_52, img_data = target_13.cuda(), target_26.cuda(), target_52.cuda(), img_data.cuda()
            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13.float(), target_13.float(), 0.3)
            loss_26 = loss_fn(output_26.float(), target_26.float(), 0.3)
            loss_52 = loss_fn(output_52.float(), target_52.float(), 0.3)

            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            times += 1
            if times % 100 == 0:
                print(times, ":  ", loss.item())
                torch.save(net.state_dict(), yolov3_net_pt_name)
                time_end = time.time()
                print("保存成功，耗时：{} 秒".format(round(time_end - time_start, 2)))