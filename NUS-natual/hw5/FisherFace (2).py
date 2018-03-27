import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator
import matplotlib.pyplot as plt

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x计算x的范数
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)每一列对应的范数

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector随机向量
    # W: D by K matrix whose columns are the principal components in decreasing order列按降序排列
    # m: mean of each projection每个投影的平均值
    classLabels = np.unique(Labels)#Labels中的不同的值
    classNum = len(classLabels)#长度为10
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]#分块，一组图片第一个的编号
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]#求平均

    #compute the within-class scatter matrix计算内部散点矩阵
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)#求协方差 Sw

    #compute the between-class scatter matrix计算相互的散点矩阵
    B = np.zeros((dim,dim))#创建一个元素全为0的矩阵
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size#求分离度 Sb

    #solve the generalized eigenvalue problem for discriminant directions为判断方向解决特征值问题
    ew, ev = linalg.eig(B, W)

    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)#既遍历编号也遍历数组
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.array(Centers).T
    return LDAW, Centers, classLabels

def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    #主要成分按降序排列的D*K矩阵
    # LL: eigenvalues特征值
    # m: mean of columns of A每一列的平均值

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column每一列减去平均值
    [r,c] = A.shape
    m = np.mean(A,1)#1就是求每一行的平均数，结果为一行
    A = A - np.tile(m, (c,1)).T#相当于将m在行方向上重复c次，列方向上重复一次
    B = np.dot(A.T, A)#协方差矩阵
    [d,v] = linalg.eig(B)#求特征值和特征向量

    # sort d in descending order按降序排列
    order_index = np.argsort(d)#排序
    order_index =  order_index[::-1]
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix计算散射矩阵的特征向量
    W = np.dot(A,v)#求矩阵点乘
    Wnorm = ComputeNorm(W)#W每一列的范数

    W1 = np.tile(Wnorm, (r, 1))#重复r行
    W2 = W / W1
    
    LL = d[0:-1]#特征值

    W = W2[:,0:-1]      #omit last column, which is the nullspace省略最后一列
    
    return W, LL, m

def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix将图片信息用矩阵表示
    # faces: face matrix in which each colummn is a colummn vector for 1 face image每列是一个图像的列向量
    # idLabels: corresponding ids for face matrix对应的id

    A = []  # A will store list of image vectors存储图像向量列表
    Label = [] # Label will store list of identity label存储身份标签
 
    # browsing the directory浏览目录
    for f in os.listdir(directory):
        if not f[-3:] =='bmp':
            continue
        infile = os.path.join(directory, f)
        #print(infile)
        im = cv2.imread(infile, 0)#以灰度模式读入,为什么im为none
        # turn an array into vector将数组变为向量
        im_vec = np.reshape(im, -1)#变为1行，自动计算长度
        A.append(im_vec)#加到A上
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A, dtype=np.float32)#变成浮点数
    faces = faces.T
    idLabel = np.array(Label)

    return faces,idLabel

def float2uint8(arr):#将arr每一个元素都相应的转化为0-255的整数
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr-mmin)/(mmax-mmin)*255
    arr = np.uint8(arr)
    return arr

if __name__=='__main__':
    faces,idLabel=read_faces(r"D:\face\train")#加r表示原始字符串，不会进行转义
    #faces=float2uint8(faces)
    W, LL, m=myPCA(faces)
    K1=90
    W1=W[:,:K1]#PCA的特征向量
    X0=[]
    Z=[]
    We = W[:,:30]
    confusion=np.zeros((10,10))
    [r,c] = faces.shape
    X0 = faces - np.tile(m, (c,1)).T#用原始数据减去平均值
    X = np.dot(W1.T,X0)#转化后的Y,K1*120
    LDAW, Centers, classLabels=myLDA(X,idLabel)
    Y = np.dot(W1.T,X0)
    Y = np.dot(LDAW.T,Y)
    
    YPLA = np.dot(We.T,X0)#转化后的Y
    #y = np.dot(We.T,x-m)
    for i in range(10):
        j=i*12+12
        z1=YPLA[:,i*12:j]
        z2=z1.mean(axis=1)
        Z.extend(z2)
    result=np.reshape(Z,(10,30))
    result=result.T
###############################################################################
    faces2,idLabel2=read_faces(r"D:\face\test")#加r表示原始字符串，不会进行转义
    W2, LL2, m2=myPCA(faces2)
    We2=W2[:,:K1]#PCA的特征向量
    Y12=[]
    Z2=[]
    [r2,c2] = faces2.shape
    Y12 = faces2 - np.tile(m, (c2,1)).T#用原始数据减去平均值
   # Y2 = np.dot(We.T,Y12)#转化后的Y2,得到test中的图像信息
    #LDAW2, Centers2, classLabels2=myLDA(Y2,idLabel2)
    Y3 = np.dot(W1.T,Y12)
    Y3 = np.dot(LDAW.T,Y3)
    Y4 = np.dot(We.T,Y12)
    a=0.9
    Y5 = np.vstack((a*Y4,(1-a)*Y3))
    compare = np.vstack((result,Centers))
    for i in range(10):
        for j in range(12):#实际编号为i
            distmin = np.linalg.norm(Y5[:,12*i+j] - compare[:,0])
            hao=0
            for f in range(10):
                dist = np.linalg.norm(Y5[:,12*i+j] - compare[:,f])
                if(dist < distmin):
                    distmin=dist
                    hao=f
            confusion[hao][i]+=1
    print(np.trace(confusion))
    print(confusion)
    

