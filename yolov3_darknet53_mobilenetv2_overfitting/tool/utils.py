import torch
from PIL import Image

def make_image_data(path):
    img=Image.open(path)
    w,h=img.size[0],img.size[1]
    temp=max(h,w)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    return mask

def iou(box, boxes, mode="inter"):
    cx, cy, w, h = box[1], box[2], box[3], box[4]
    cxs, cys, ws, hs = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]

    box_area = w * h # 最小面积
    boxes_area = ws * hs # 最大面积

    _x1, _x2, _y1, _y2 = cx - w/2, cx + w/2, cy - h/2, cy + h/2
    _xx1, _xx2, _yy1, _yy2 = cxs - ws / 2, cxs + ws / 2, cys - hs / 2, cys + hs / 2

    xx1 = torch.maximum(_x1, _xx1) # 左上角   最大值
    yy1 = torch.maximum(_y1, _yy1) # 左上角   最大值
    xx2 = torch.minimum(_x2, _xx2) # 右下角  最小值
    yy2 = torch.minimum(_y2, _yy2) # 右下角  最小值

    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    w = torch.clamp(xx2 - xx1, min=0) # ★夹
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h

    if mode == 'inter':
        return inter / (box_area + boxes_area - inter) #交集除以并集
    elif mode == 'min':
        return inter / torch.min(box_area, boxes_area)

# yolo多类别目标检测中，nms的每轮抑制时 对不同类别不做抑制
def nms(boxes, thresh, mode='inter'):
    args = boxes[:, 5].argsort(descending=True)  # 排序,最大的再第一个
    sort_boxes = boxes[args]  # 排序后的 框


    keep_boxes = []
    while len(sort_boxes) > 0:
        _box = sort_boxes[0]  # 最大框
        _cls = _box[6]
        _cx_index, _cy_index = _box[7], _box[8]

        keep_boxes.append(_box)  # 装入最大框

        if len(sort_boxes) > 1:
            _boxes = sort_boxes[1:]  # 其他框
            _cx_indexes, _cy_indexes = _boxes[:,7], _boxes[:,8]
            _cx_index_mask, _cy_index_mask = _cx_index == _cx_indexes, _cy_index == _cy_indexes
            _cx_cy_mask = _cx_index_mask & _cy_index_mask # 相同中心点掩码

            _clses = _boxes[:, 6]
            _mask = _cls != _clses # 不同类别 掩码

            _iou = iou(_box, _boxes, mode)

            if False:
                #！第一种写法
                _iou_mask = _iou < thresh # iou 掩码
                # 同一个中心点的情况下，保留不同类别 且满足iou的
                # 保留不同中心点且满足iou的（不同的中心点可以保证不是同一个目标）
                sort_boxes = _boxes[(_mask & _iou_mask) | (~_cx_cy_mask & _iou_mask)] # ok
                # 实际上这里做复杂了, 从保留的角度去考虑问题了。
                # 简单的做法应该是从 _iou_mask 掩码入手,改成 _iou > thresh，
                # 再将这个 _iou_mask 中不同类别的去掉， 此时的 _iou_mask 是要去重的 mask，
                # 再对 _iou_mask 取反，就可以的得到要保留的 _iou_mask 了，
                # 然后 sort_boxes = _boxes[_iou_mask] 就可以了
            else:
                #！ 第二种写法
                # 最终表达含义就是,在(非iou保留框中-要保留的其他类的框),再整体取反就解决了
                _iou_mask = _iou > thresh
                sort_boxes = _boxes[~(_iou_mask & ~_mask)]
            
        else:
            break
    return keep_boxes


# def detect(feature_map, thresh):
#     masks = feature_map[:, 4, :, :] > thresh
#     idxs = torch.nonzero(masks)


if __name__ == '__main__':
    box = torch.Tensor([2, 2, 3, 3, 6])
    boxes = torch.Tensor([[2, 2, 3, 3, 6], [2, 2, 4, 4, 5], [2, 2, 5, 5, 4]])
    print(iou(box, boxes, mode="inter"))
    print(nms(boxes, 0.1))

    # import numpy as np
    #
    # a = np.array([[1, 2], [3, 4]])
    # print(a[:, 1])
