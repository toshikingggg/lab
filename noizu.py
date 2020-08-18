import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
検査領域200<x<1000
       400<y<700
ノイズ除去範囲

"""

img_src1 = cv2.imread("/Volumes/Transcend/卒業研究/program/testryuryo/musyoku/img1.tif") #背景画像


a = input("読み込む画像の枚数:")
a = int(a)
src_size = 0
for i in range(1, a):
     img_src2 = cv2.imread("/Volumes/Transcend/卒業研究/program/testryuryo/musyoku/img%d.tif"%(i))
     #img_src2 = cv2.imread("/Volumes/Transcend/卒業研究/program/testryuryo/testimg/img{i}.bmp") #背景画像全て読み込み
     src_size +=1
     """
     cv2.namedWindow('tyusyutu', cv2.WINDOW_NORMAL)
     cv2.imshow('tyusyutu',img_src2)
     cv2.waitKey(0)
     """

     img_diff = cv2.absdiff(img_src2, img_src1) #差分画像
     img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY) #グレースケール化
     blur = cv2.GaussianBlur(img_diff,(5,5),0) #ガウシアンフィルタ
     img_diffm = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1] #マスク画像作成、大津の二値化
     """
     cv2.namedWindow('img', cv2.WINDOW_NORMAL)
     cv2.imshow('img',img_diffm)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     """
     """  
     img_diff = cv2.cvtColor(img_diff, cv2.COLOR_GRAY2RGB) 
     img_diffm = cv2.cvtColor(img_diffm, cv2.COLOR_GRAY2RGB)
     plt.subplot(121)
     plt.imshow(img_diff)
     plt.title("diff image")
     plt.subplot(122)
     plt.imshow(img_diffm)
     plt.title("mask")
     plt.show()
     """

 
     #ノイズ画像除去も行う
     cnt = 0
     for x in range(500, 550):
         for y in range(300,400):
             if img_diffm[y,x] == 255:
                 cnt += 1 #ここまでノイズ除去検査領域外を調べる
                 if cnt > 5:
                     p = 0
                     if p == 0:
                         lists = []
                         lists.append(src_size)
         cnt = 0
l = max(lists)  
print("ノイズが含まれる画像の枚数:",l)