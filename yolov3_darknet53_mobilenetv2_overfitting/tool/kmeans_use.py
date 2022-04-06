import glob
import xml.etree.cElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou


# ANNOTATIONS_PATH = "E:\MyData\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations"
ANNOTATIONS_PATH = r"../VOC2007/Annotations"
CLUSTERS = 9


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        length = width if width > height else height

        try:
            for obj in tree.iter("object"):
                xmin = int(obj.findtext("bndbox/xmin")) / length
                ymin = int(obj.findtext("bndbox/ymin")) / length
                xmax = int(obj.findtext("bndbox/xmax")) / length
                ymax = int(obj.findtext("bndbox/ymax")) / length

                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                if xmax == xmin or ymax == ymin:
                    print(xml_file)
                dataset.append([xmax - xmin, ymax - ymin])
        except:
            print(xml_file)
    return np.array(dataset)


if __name__ == '__main__':
    # print(__file__)
    data = load_dataset(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    # clusters = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    # out= np.array(clusters)/416.0
    print(out)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    ws = (out[:,0] * 416).astype(np.int)
    ys = (out[:,1] * 416).astype(np.int)
    print("Boxes:\n {}-{}".format(ws, ys))

    suggest_boxes = {}
    for i in range(9):
        suggest_boxes[ws[i] * ys[i]] = (ws[i], ys[i])

    print(suggest_boxes)
    for i in sorted (suggest_boxes.keys()) :
        print (i, ":", suggest_boxes[i])


    # ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    # print("Ratios:\n {}".format(sorted(ratios)))

