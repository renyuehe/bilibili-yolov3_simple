# 使用哪个主干网络
is_use_mobilenetv2 = False

# 使用哪个权重
if is_use_mobilenetv2:
    yolov3_net_pt_name = r'net/overfit_mobilenetv2_net.pt'
    yolov3_merge_name = r"yolov3_mobilenetv2_overfitting_mergeimg.jpg"
else:
    yolov3_net_pt_name = r"net/overfit_darknet53_net.pt"
    yolov3_merge_name = r"yolov3_darknet53_overfitting_mergeimg.jpg"