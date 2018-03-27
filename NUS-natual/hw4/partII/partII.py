import cv2
import os

def detect_face(img_name):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#default为轻量级检测，容易出现失误，可以使用alt，alt2
	# 'haarcascade_frontalface_default.xml' is provided by opencv, you can find it ine the opencv folder 
	img = cv2.imread(img_name)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#将图片转化为灰度图
	faces = face_cascade.detectMultiScale(gray)#判断该灰度图是否是脸
	if not len(faces):
		print('no face detected for', img_name)
	else:
		print(len(faces), 'faces detected for', img_name)
	
		for (x, y, w, h) in faces:#返回值faces为人脸坐标和大小，w为宽，h为长
			cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0),2)#绘制矩形，分别为图片，两个关键坐标，颜色，线的类型

		cv2.imwrite(img_name[:-4]+'_result'+img_name[-4:], img)#命名，前四后四

if __name__=='__main__':
	#write your code here
    os.chdir("C:\\Users\魏硕\Desktop\hw4\partII")
    detect_face('test14.png')