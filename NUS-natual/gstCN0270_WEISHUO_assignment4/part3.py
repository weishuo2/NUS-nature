# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:13:38 2017

@author: 魏硕
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\魏硕\Desktop\hw4")

P=np.mat([[2],[3],[1]])
C=np.mat([[3],[2]])
T1=np.mat([[1,0,-C[0][0]],[0,1,-C[1][0]],[0,0,1]])
T2=np.mat([[1,0,C[0][0]],[0,1,C[1][0]],[0,0,1]])
for i in range(1,8):
    k=math.pi * (i/4)
    R=np.mat([[math.cos(k),-math.sin(k),0],
              [math.sin(k),math.cos(k),0 ],
              [0          ,          0,1]])
    result1=np.dot(T1,P)#将原点变为C后，P的坐标
    p1,=plt.plot(result1[0][0],result1[1][0],'ro',label="Trans1")
    result2=np.dot(R,result1)#旋转角度k
    p2,=plt.plot(result2[0][0],result2[1][0],'bo',label="Rotate")
    result3=np.dot(T2,result2)#将原点变回原来的原点
    p3,=plt.plot(result3[0][0],result3[1][0],'ko',label="Trans2")
plt.legend(handles = [p1, p2, p3,], labels = ['Trans1', 'Rotate','Trans2'])
plt.savefig("3_result")#设置备注
plt.show()
 
