
# 图片的宽高
IMG_HEIGHT = 416
IMG_WIDTH = 416

# 分类数
CLASS_NUM = 3

VOC_CLASSES = (
    'person',
    'horse',
    'bicycle'
)

# 9个建议框
ANCHORS_GROUP = {
    13: [[301, 184], [303, 215], [217, 312]],
    26: [[132, 107], [121, 249], [230, 155]],
    52: [[36, 19], [51, 36], [65, 138]]
}

# 684 : (36, 19)
# 1836 : (51, 36)
# 8970 : (65, 138)
# 14124 : (132, 107)
# 30129 : (121, 249)
# 35650 : (230, 155)
# 55384 : (301, 184)
# 65145 : (303, 215)
# 67704 : (217, 312)

# 9个框建议框分别对应的面积
ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}

if __name__ == '__main__':
    for iter in ANCHORS_GROUP.items():
        print(iter)

    print()
    for iter in ANCHORS_GROUP_AREA.items():
        print(iter)
