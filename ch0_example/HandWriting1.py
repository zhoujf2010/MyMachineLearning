# -*- coding:utf-8 -*-
'''
Created on 2017年12月14日

@author: zjf
'''

'''
手写数字识别
'''

import numpy as np;
import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from tkinter import Frame,Button,Canvas,Label
from sklearn.linear_model import LogisticRegression 
from PIL import Image,ImageDraw
from sklearn.preprocessing import label_binarize
import warnings


class MainFrame(Frame):  # 定义窗体

    def __init__(self, mode, master=None):
        Frame.__init__(self, master)
        self.createWidgets()
        self.place()
        self.master.title('手写测试')
        left = (self.master.winfo_screenwidth() - 150) / 2
        top = (self.master.winfo_screenheight() - 300) / 2
        self.master.geometry("150x250+%d+%d" % (left, top))  
        self.master.resizable(0, 0)  # 去最大化
        self.isdown = False
        self.oldPos = []
        self.img = Image.new("RGB", (128, 128), (255, 255, 255))
        self.mode = mode

    def createWidgets(self):
        self.can = can = Canvas(self.master, width=128, height=128, bg="#FFFFFF")
        can.bind("<ButtonPress-1>", self.mousedown)
        can.bind("<ButtonRelease-1>", self.mouseup)
        can.bind("<Motion>", self.movemouse)
        can.place(x=10, y=10)
        
        self.lbl = lbl = Label(self.master, text='识别结果：-')
        lbl.place(x=10, y=140)
        
        Button(self.master, text='识别', command=self.docalc).place(x=20, y=170, width=100, height=30)
        Button(self.master, text='清空', command=self.clear).place(x=20, y=210, width=100, height=30)

    def clear(self):
        self.can.delete("all")
        self.img = Image.new("RGB", (128, 128), (255, 255, 255))  # 重置图片
        self.lbl["text"] = ""
    
    def movemouse(self, event):
        if self.isdown :
            self.can.create_line((self.oldPos[0], self.oldPos[1]), (event.x, event.y), width=5)
            draw = ImageDraw.Draw(self.img)  # 在图片对象同步绘制
            draw.line([self.oldPos[0], self.oldPos[1], event.x, event.y], (0, 0, 0), width=5)
            self.oldPos = [event.x, event.y]
        
    def mousedown(self, event):
        self.isdown = True
        self.oldPos = [event.x, event.y]
        
    def mouseup(self, event):
        self.isdown = False
        self.oldPos = []
    
    def getCanvImg(self):
        self.can.update()
        return np.array(self.img)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    
    def JudgeEdge(self, img, direction, isback):
        color = 50  # 判断内容阈值
        for i in range(np.shape(img)[direction]):
            if direction == 0:
                line1 = (img[i * isback, :] > color).tolist().count(True)
            else:
                line1 = (img[:, i * isback] > color).tolist().count(True)
            if line1 > 0:  # 有内容则表示到边，跳出
                return i
        return -1

    def docalc(self):
        data = self.getCanvImg();
        print(np.shape(data))
        # 灰度处理
        data = self.rgb2gray(data)
        print(np.shape(data))
        
        # 反转值
        data = 255 - data
        # Image.fromarray(gray).convert('RGB').save('E:\\aa.png')
        width, height = np.shape(data)
        
        # 截边
        top1 = self.JudgeEdge(data, 0, 1) 
        top2 = height - self.JudgeEdge(data, 0, -1) 
        left1 = self.JudgeEdge(data, 1, 1) 
        left2 = width - self.JudgeEdge(data, 1, -1) 
        # 扩边
        top1 = 0 if top1 < 10 else top1 - 10
        top2 = height if height - top2 < 10 else top2 + 10
        left1 = 0 if left1 < 10 else left1 - 10
        left2 = width if width - left2 < 10 else left2 + 10
            
        # 调整为正方形
        if top2 - top1 > left2 - left1:
            p = (top2 - top1 - left2 + left1) / 2
            if p > left1:
                p1 = left1
                left1 = 0
                left2 += p * 2 - p1
            elif p > height - left2:
                p1 = height - left2
                left1 = p * 2 - p1
                left2 += height
            else:
                left1 -= p
                left2 += p
        elif top2 - top1 < left2 - left1:
            p = (left2 - left1 - top2 + top1) / 2
            if p > top1:
                p1 = top1
                top1 = 0
                top2 += p * 2 - p1
            elif p > width - top2:
                p1 = width - top2
                top1 = p * 2 - p1
                top2 += width
            else:
                top1 -= p
                top2 += p
            
        data = data[int(top1):int(top2), int(left1):int(left2)]
        
        # 缩放
        data = Image.fromarray(data * 16).resize((8, 8), Image.ANTIALIAS)  
        data = np.array(data)
        
        # 去除非法数字
        data[data < 0] = 0
        
        # 颜色缩为16位
        data = np.around(data / 16)
        
        # Image.fromarray(255 - data).convert('RGB').save('E:\\y.png')  # 将示例输出保存为图片
        
        # 模型预测结果
        data = data.reshape(1, -1)
        t = mode.predict(data)
        print ('data=', data, '\n识别=', t[0])
        self.lbl["text"] = t[0]
        
        
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # 去掉警告
    np.set_printoptions(suppress=True,linewidth=500) #设置展示时不要用科学计数法
    
    # 从库中加载数据
    digits = datasets.load_digits()
    print('shape=',np.shape(digits.images), np.shape(digits.target))
    print(digits.images[0],'\r\n',digits.target[0]) #输出一个示例数据
    Image.fromarray(255 - digits.images[0] * 16).convert('RGB').save('E:\\x.png') #将示例输出保存为图片
    
    # 单条数据变成一维
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    # 送模型训练
    mode = LogisticRegression()
    mode.fit(data, digits.target)
    print('0号数据预测:',mode.predict(data[0].reshape(1, -1)))
    
    # 评分
    y_hat = mode.predict(data)
    y_score = mode.decision_function(data)  # 得到每个的评分
    y = label_binarize(digits.target, classes=np.arange(10))  # one-hot编码
    
    fpr, tpr, thresholds = metrics.roc_curve(y.ravel(), y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(digits.target, y_hat)  # 计算精确度
    print('auc=', auc, u'精确度=', accuracy)
    #plt.plot(fpr, tpr, c='r', lw=2, ls='-', alpha=0.8)
    #plt.show()
    
    # 显示手写窗体
    app = MainFrame(mode)
    app.mainloop()
