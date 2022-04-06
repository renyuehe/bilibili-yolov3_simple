import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cfg
import os
from tool.utils import make_image_data

from PIL import Image
import math

LABEL_FILE_PATH = "data/voc_train_label.txt"
IMG_BASE_DIR = "data"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b

class MyDataset(Dataset):
    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        _img_data = make_image_data(os.path.join(IMG_BASE_DIR, strs[0])) # 返回填充后的正方形 img
        max_w, max_h = _img_data.size
        case = 416/max_w # 缩放比

        _img_data = _img_data.resize((416,416)) # 此处要等比缩放,对图片进行一个等比缩放
        img_data = transforms(_img_data)

        _boxes = np.array([float(x) for x in strs[1:]])
        # _boxes = np.array(list(map(float, strs[1:])))
        boxes = np.split(_boxes, len(_boxes) // 5)

        # 遍历尺寸
        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM)) # h,w,3,25

            # 遍历真实框
            for box in boxes:
                cls, cx, cy, w, h = box #真实框的中心点 和 宽高（是resize后的数据）
                cx, cy, w, h = cx * case, cy * case, w * case, h * case # 对等比例

                cx_offset, cx_index = math.modf(cx / (cfg.IMG_WIDTH / feature_size))
                cy_offset, cy_index = math.modf(cy / (cfg.IMG_WIDTH / feature_size))

                # 遍历形状
                for i, anchor in enumerate(anchors):

                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    p_area = w * h
                    iou = min(p_area, anchor_area * case**2) / max(p_area, anchor_area * case**2) # ★相似度就是置信度

                    if labels[feature_size][int(cy_index), int(cx_index), i][0] < iou: # 重复的选置信度最大的
                        labels[feature_size][int(cy_index), int(cx_index), i] = np.array([iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h),
                                                                                          *one_hot(cfg.CLASS_NUM, int(cls))])

        return labels[13], labels[26], labels[52], img_data


from PIL import Image, ImageDraw
if __name__ == '__main__':
    data = MyDataset()

    print(data[0][0].shape)
    print(data[0][1].shape)
    print(data[0][2].shape)
    print(data[0][3].shape)

    print(data[0][0][...,0].shape)

    ret = data[0][0][..., 1:5]
    print(ret)
    print(ret[ret != 0])

