import os
from functools import reduce
import xml.etree.cElementTree as ET

from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

from tool.utils import make_image_data
from cfg import VOC_CLASSES

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    '''
    解决cv2绘制中文字体问题,并且中文字配置对应的矩形框
    '''
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    position_left_up = (position[0], position[1] - textSize)
    position_right_down = (position[0] + (textSize * (3+len(text))) / 2,  position[1])
    # draw.rectangle((*position_left_up, *position_right_down),fill="red", width=2)

    draw.text(position_left_up, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def get_xml_info(xml_filepath):
    tree = ET.parse(xml_filepath)

    filename = tree.findtext("./filename")
    height = int(tree.findtext("./size/height"))
    width = int(tree.findtext("./size/width"))

    labels_record = []
    for obj in tree.iter("object"):

        xmin = int(obj.findtext("bndbox/xmin"))
        ymin = int(obj.findtext("bndbox/ymin"))
        xmax = int(obj.findtext("bndbox/xmax"))
        ymax = int(obj.findtext("bndbox/ymax"))

        cx = (xmin + xmax)//2      # 中心点 x
        cy = (ymin + ymax)//2      # 中心点 y
        w = (xmax - xmin)          # 宽
        h = (ymax - ymin)          # 高

        cx, cy, w, h = int(cx), int(cy), int(w), int(h) # 做标签时候保持原来的位置, 因为非正方形图片只会在右侧或下侧填充
        print(cx, cy, w, h)
        print(VOC_CLASSES.index(obj.findtext("name")))
        try:
            labels_record.append([VOC_CLASSES.index(obj.findtext("name")), cx, cy, w, h])
        except Exception as E:
            raise E

    # print(labels_record)
    record_b = [reduce(lambda x, y: str(x) + " " + str(y), i) for i in labels_record]  # 使用 lambda 匿名函数
    # print(record_b)
    return filename +' ' + ' '.join(record_b) + '\n'

def generate_labels(rootpath, dstpath):
    dstfile = open(dstpath, 'a+')

    Anno_path = "Annotations"
    Anno_path = os.path.join(rootpath, Anno_path)
    Jpg_path = "JPEGImages"
    Jpg_path = os.path.join(rootpath, Jpg_path)

    for file in os.listdir(Anno_path):
        xml_file_path = os.path.join(Anno_path, file)
        try:
            records = get_xml_info(xml_file_path)
        except Exception as E:
            continue
        dstfile.write(Jpg_path + '\\' + records)


import cv2
if __name__ == '__main__':

    # ######## 生成标签 #########
    generate_labels(r"../VOC2007",
                    r"..\data\voc_train_label.txt")
    generate_labels(r"../VOC2007",
                    r"..\data\voc_test_label.txt")

    ######### 查看标签是否正确 #########
    # img = Image.open(r"../VOC2007\JPEGImages\000017.jpg")

    img = make_image_data(r"../VOC2007\JPEGImages\000017.jpg")
    img.show()

    # img = img.resize((416, 416))
    draw = ImageDraw.Draw(img)

    file = open(r"../data/voc_train_label.txt", "r")
    firstline = file.readline()
    strlist = firstline.split()
    filepath = strlist[0]
    boxes = strlist[1:]
    print(filepath)
    print(boxes)

    for i in range(0,14, 5):
        cls = boxes[i]
        cx = int(boxes[i+1])
        cy = int(boxes[i+2])
        w = int(boxes[i+3])
        h = int(boxes[i+4])
        print("1 >>>>", cls, cx, cy, w, h)

        draw.rectangle(xy=[(cx-w//2, cy-h//2), (cx+w//2, cy+h//2)], fill="red", width=1)
        break

    img.show()
    cv2.waitKey(0)

    # ############# 测试voc本身的数据集 ##############
    # img = Image.open(r"../VOC2007\JPEGImages\000017.jpg")
    # draw = ImageDraw.Draw(img)
    #
    # draw.rectangle(xy=[(232 - 94 // 2, 130 - 137 // 2), (232 + 94 // 2, 130 + 137 // 2)], fill="red", width=1)
    #
    # img.show()
    # cv2.waitKey(0)



