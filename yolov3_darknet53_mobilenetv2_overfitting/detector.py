import PIL.Image

from modules.yolov3 import *
import os
import torch
from tool import tools
import cfg
from PIL import Image, ImageDraw, ImageFont
import cv2
from tool.utils import nms,make_image_data
from dataset import transforms

import common
# 侦测
class Detector(torch.nn.Module):

    def __init__(self, net_filepath):
        super(Detector, self).__init__()

        self.net = Yolov3(use_mobilenetv2=is_use_mobilenetv2)
        if os.path.exists(net_filepath):
            try:
                self.net.load_state_dict(torch.load(net_filepath))
                print("加载先验成功")
            except Exception as E:
                print("加载权重失败")
                print(E)
        self.net.eval() # 模型中用了 batchnormal 这里必须要调用 eval()

    def forward(self, input, thresh, anchors, case):

        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13], case)#idxs_13索引，vecs_13偏移量，13x13的缩放比例是32

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26], case)

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52], case)

        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

        boxes = nms(boxes, 0.7, mode="inter")

        return boxes

    def _filter(self, output, thresh):
        # n,c,h,w ==>> n,h,w,c
        output = output.permute(0, 2, 3, 1)

        # n,h,w,c ==>> n,h,w,3,8
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)

        #output:N,H,W,3,8 ==>> mask:N,H,W,3
        mask = torch.sigmoid(output[..., 0]) > thresh # 拿到阈值

        # idxs ==>> N,4
        idxs = mask.nonzero()

        # vecs ==>> N,8
        vecs = output[mask]

        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors, case):
        # idxs 索引、   N,4 （N,H,W,C）
        # vecs 偏移量、 N,15
        # t 缩放比例
        # anchors 建议框
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框
        cx_index = idxs[:, 1]
        cy_index = idxs[:, 2]

        # 拿到iou置信度
        cfd = torch.sigmoid(vecs[:,0])

        # case 是图像整体的缩放比
        # 中心点反算： (索引+偏移量)*缩放比例
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t / case  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t / case  # 原图的中心点x

        # case 是图像整体的缩放比
        # 宽高偏移量标签：log(实际框w / 建议框w) = log(偏移量offset)
        # 宽高偏移量反算：建议框w * e^(偏移量offset)
        w = anchors[a, 0] * torch.exp(vecs[:, 3]) / case
        h = anchors[a, 1] * torch.exp(vecs[:, 4]) / case

        # 类别
        data = vecs[:, 5:]
        _, max = torch.max(data, dim=1, keepdim=True)
        max = torch.squeeze(max,dim=1) # 这里要指定对 第一个维度降维, 如果不指定可能会降多个维度

        return torch.stack([n.float(), cx, cy, w, h, cfd, max, cx_index, cy_index], dim=1)

from common import *
from tool.PIL_Merge_image import image_merge
imglist = []
if __name__ == '__main__':

    detector = Detector(yolov3_net_pt_name)

    # imgfilename = r"VOC2007/JPEGImages/001370.jpg"
    path = r"VOC2007/JPEGImages"
    for imgName in os.listdir(path):
        imgfilename = os.path.join(path, imgName)
        img = make_image_data(imgfilename)
        w, h = img.size
        case = 416/w
        img = img.resize((416,416))

        batch_img = transforms(img)[None,...]
        y2 = detector(batch_img, 0.2, cfg.ANCHORS_GROUP, case) # 侦测
        print(imgName,">>>>>>>>>>>>>",y2)

        img = PIL.Image.open(imgfilename)
        for i, box in enumerate(y2):
            cx, cy, w, h, conf, cls = int(box[1]), int(box[2]), int(box[3]), int(box[4]), float(box[5]), box[6]
            xmin, ymin, xmax, ymax = cx-w//2, cy-h//2, cx+w//2, cy+h//2

            draw = ImageDraw.Draw(img)
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=1)

            try:#防止类别出错时候超出了VOC_CLASSES的范围
                # font1 = ImageFont.truetype(r'C:\Windows\Fonts\ARLRDBD.TTF', 20)
                draw.text((xmin, ymin), f"{round(conf,2)} : {tools.VOC_CLASSES[int(cls)]}")
            except:
                continue

        img.show()
        cv2.waitKey(0)
        # imglist.append(img)
        # image_merge(imglist, output_dir="D:/Desktop/yolov3_imgsave", output_name=yolov3_merge_name)
