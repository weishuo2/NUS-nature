# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:36:04 2017

@author: 魏硕
"""

	
from PIL import Image,ImageOps
import numpy
from numpy import linalg
#import matplotlib 
from matplotlib import pyplot
import os 
os . chdir ("C:\\Users\魏硕\Desktop\hw3 ")
#读取图像
flower = Image.open("flower.bmp")
flower.show()
#######################
flower = ImageOps.grayscale(flower) #转化为灰度图像
aflower = numpy.asarray(flower) # aflower is unit8 
aflower = numpy.float32(aflower)#转化格式
U,S,Vt = linalg.svd(aflower)#奇异值分解
pyplot.plot(S,'b.') 
pyplot.savefig("5_result1")
pyplot.show()
################
K = 200 #？？？？不太懂
Sk = numpy.diag(S[:K]) 
Uk = U[:, :K] 
Vtk = Vt[:K, :]#提取出前二十个奇异值
######################
aImk = numpy.dot(Uk, numpy.dot( Sk, Vtk)) #将提取出来的数据进行合并
Imk = Image.fromarray(aImk)
pyplot.imshow(Imk)#Imk.show()这样显示怎么保存？
pyplot.axis("off")#避免显示坐标轴
pyplot.savefig("5_result5")
Imk.show()
Imk.save("5_result5.tiff")
