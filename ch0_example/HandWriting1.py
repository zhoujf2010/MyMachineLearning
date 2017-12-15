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
from sklearn import datasets, svm, metrics
from Tkinter import *
# from Tkinter.ttk import *
import random


class MainFrame(Frame):  # 定义窗体

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.createWidgets()
        self.place()
        self.master.title('手写测试')
        left = (self.master.winfo_screenwidth() - 150) / 2
        top = (self.master.winfo_screenheight() - 300) / 2
        self.master.geometry("150x250+%d+%d" % (left, top))  
        self.master.resizable(0, 0) #去最大化
        self.isdown = False
        self.oldPos = []

    def createWidgets(self):
        self.can = can = Canvas(self.master, width=128, height=128, bg="#FFFFFF")
        can.bind("<ButtonPress-1>", self.mousedown)
        can.bind("<ButtonRelease-1>", self.mouseup)
        can.bind("<Motion>", self.movemouse)
        can.place(x=10, y=10)
        
        lbl = Label(self.master, text='识别结果：-')
        lbl.place(x=10, y=140)
        
        Button(self.master, text='识别', command=self.docalc).place(x=20, y=170, width=100, height=30)
        Button(self.master, text='清空', command=self.clear).place(x=20, y=210, width=100, height=30)
    
    def docalc(self):
        self.quit()
        
    def clear(self):
        self.can.delete("all")
    
    def movemouse(self, event):
        if self.isdown :
            # self.can.create_oval(event.x, event.y, event.x + 1, event.y + 1, fill="black")
            self.can.create_line((self.oldPos[0], self.oldPos[1]), (event.x, event.y), width=5)
            self.oldPos = [event.x, event.y]
        
    def mousedown(self, event):
        self.isdown = True
        self.oldPos = [event.x, event.y]
        
    def mouseup(self, event):
        self.isdown = False
        self.oldPos = []
    
    def getCanvImg(self):
        return []
# https://stackoverflow.com/questions/9886274/how-can-i-convert-canvas-content-to-an-image


if __name__ == '__main__':
#     digits = datasets.load_digits()
#     print np.shape(digits.images)
#     print np.shape(digits.target)
#     print digits.images[1796]
#     #print digits.data[1796]
#     print digits.target[1796]
#     print digits.DESCR
# 
#     images_and_labels = list(zip(digits.images, digits.target))
#     for index, (image, label) in enumerate(images_and_labels[:4]):
#         plt.subplot(2, 4, index + 1)
#         plt.axis('off')
#         plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#         plt.title('Training: %i' % label)

    app = MainFrame()
    app.mainloop()

