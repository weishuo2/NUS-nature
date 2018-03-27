# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 22:50:46 2017

@author: 魏硕
"""

from PIL import Image,ImageFilter
from scipy import ndimage
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\魏硕\Desktop\hw3")
img1=Image.open('5_result1.png')
img2=Image.open('5_result2.png')
img3=Image.open('5_result3.png')
img4=Image.open('5_result4.png')
img5=Image.open('5_result5.png')
img6=Image.open('flower.bmp')
plt.subplot(2,3,1)
plt.imshow(img1)
plt.subplot(2,3,2)
plt.imshow(img2)
plt.axis('off') 
plt.subplot(2,3,3)
plt.imshow(img3)
plt.axis('off') 
plt.subplot(2,3,4)
plt.imshow(img4)
plt.axis('off') 
plt.subplot(2,3,5)
plt.imshow(img5)
plt.axis('off') 
plt.subplot(2,3,6)
plt.imshow(img6)
plt.axis('off') 
plt.savefig("5_result")