'''
Created on 2020年10月16日

@author: zjf

使用YoloV4 实现视频中的目标检测
'''

from tool.darknet2pytorch import Darknet
import cv2
import numpy as np
import math
import time
import torch as torch
import tool.utils as utils
import matplotlib.pyplot as plt
import os
import wget
import subprocess

use_cuda = False


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


# 随机设定边线色
def get_color(x, max_val):
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][2] + ratio * colors[j][2]
    g = (1 - ratio) * colors[i][1] + ratio * colors[j][1]
    b = (1 - ratio) * colors[i][0] + ratio * colors[j][0]
    return (int(r * 255), int(g * 255), int(b * 255))


def loadModel(cfgfile, weightfile):
    model = Darknet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    if use_cuda:
        model.cuda()
    model.eval()
    return model


def predect(model, img, conf_thresh, nms_thresh):
#     img = cv2.imread('datasets/dog.jpg')
    img = cv2.resize(img, (model.width, model.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    if use_cuda:
        img2 = img2.cuda()
    img2 = torch.autograd.Variable(img2)
    output = model(img2)
    boxes = utils.post_processing(img, conf_thresh, nms_thresh, output)
    return boxes[0]


def drawImgBox(img, boxes, class_names):
    width = img.shape[1]
    height = img.shape[0]
    for box in boxes:
        pt1 = (int(box[0] * width), int(box[1] * height))
        pt2 = (int(box[2] * width), int(box[3] * height))
        cls_conf = box[5]
        cls_id = box[6]

        print('%s: %f' % (class_names[cls_id], cls_conf))
        
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        rgb = get_color(offset, classes)
        img = cv2.putText(img, class_names[cls_id], pt1, cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
        img = cv2.rectangle(img, pt1, pt2, rgb, 1)
    return img

# wget 


if __name__ == '__main__':
    #注意，需要放置一个ffmpeg.exe在文件夹下
    if not os.path.exists("datasets/yolov4.weights"):
        wget.download("https://pjreddie.com/media/files/yolov3.weights", 'datasets/yolov4.weights')
    
    model = loadModel("datasets/yolov4.cfg", "datasets/yolov4.weights")

    print('classnum=', model.num_classes)
    class_names = load_class_names('datasets/label.txt')

    in_rtsp = "rtsp://ceshi:epoint123@192.168.191.2:554/Streaming/Channels/101?transportmode=unicast"
    out_rtmp = "rtmp://192.168.186.65:1935/live/test"
    
    cap = cv2.VideoCapture(in_rtsp)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    sizeStr = str(frame_size[0]) + 'x' + str(frame_size[1])
    outputrate = 10
    command = ['ffmpeg', '-y', '-an', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', sizeStr, '-r', str(outputrate), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast', '-f', 'flv', out_rtmp]
    pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    

        boxes = predect(model, frame, 0.4, 0.6)
        frame = drawImgBox(frame, boxes, class_names)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        pipe.stdin.write(frame.tostring())
        
    cap.release()
    pipe.terminate()