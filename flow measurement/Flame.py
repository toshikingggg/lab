import cv2
import numpy as np
import noizu
"""
a=1の時、基準点x=200を超えた時のフレーム数を求める。
a=2の時、基準点x=1200を超えた時のフレーム数を求める。
a=3の時、流量を算出する。
"""

img_src1 = cv2.imread("/Volumes/Transcend/卒業研究/program/testryuryo/musyoku/img1.tif") #背景画像

a = noizu.l + 1 #input("ノイズが含まれない最小の画像の数:")
b = noizu.a #input("読み込む画像の枚数")
#a = int(a)
#b = int(b)
src_size = a - 1

flag = False

for i in range(a, b):
     img_src2 = cv2.imread("/Volumes/Transcend/卒業研究/program/testryuryo/musyoku/img%d.tif"%(i))
     #img_src2 = cv2.imread("/Volumes/Transcend/卒業研究/program/testryuryo/testimg/img{i}.bmp") #背景画像全て読み込み
     src_size +=1
     """cv2.namedWindow('tyusyutu', cv2.WINDOW_NORMAL)
     cv2.imshow('tyusyutu',img_src2)
     cv2.waitKey(0)"""

     img_diff = cv2.absdiff(img_src2, img_src1) #差分画像
     img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY) #グレースケール化
     blur = cv2.GaussianBlur(img_diff,(5,5),0) #ガウシアンフィルタ
     img_diffm = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1] #マスク画像作成、大津の二値化
     """cv2.namedWindow('img', cv2.WINDOW_NORMAL)
     cv2.imshow('img',img_diffm)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     """

                         
     #流量算出
 
     #定義(最小フレーム数)、x座標が380を超えた時
     #白点の数をカウント
     cnt = 0
     
     finish =False
     for x in range(200, 1280) :
         for y in range(400, 700):
             if img_diffm[y,x] == 255:
                 cnt += 1
                 if cnt > 10:  # 閾値を超えたら配列の中にそのx（座標）の値を保存していく
                     lists = []
                     lists.append(x)
                     cnt = 0
                     if flag == False:
                         if min(lists) == 200: # x座標が380を超えた時のフレーム数を確かめる
                             flag = True
                             k = src_size
                             print("フレーム数の最小値:",k)
                     if min(lists) == 1200:
                         finish = True
                         n = src_size
                         print("フレーム数の最大値:",n)
                         break
                     if finish == True:
                         break
                 if finish == True:
                     break
             if finish == True:
                 break
         if finish == True:
             break
     if finish == True:
         break