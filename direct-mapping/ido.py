import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



"""
多項式を使ってx',y'の値を計算する
x'=a,y'=bとする。係数はf2,f1の順で使う(x,y軸反転のため)
"""


img = cv2.imread('図1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


img2 = np.ones((1024,1280),np.uint8)*255
plt.plot(), plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.show()

#DT = [] #計算後のデータ格納用。
#DT2 = []
#img(y,x)の順になるから逆で計算をする。
for x in range(0,860):
    for  y in range(0,820):
            px = img[x,y] #輝度値
            b = -380.3311 + 2.0868*x + 5.1813*y + (-0.0016)*x**2 + -0.0039*y**2 + -0.0129*x*y + 2.72e-06*x**3 +-3.067e-06*y**3 + -4.268e-06*y*x**2 + (1.827e-05)*x*y**2
            a = -82.2328 + 1.6399*x + 0.4374*y + (-0.0014)*x**2 + (-0.0009)*y**2 + (-0.0007)*x*y + (8.795e-07)*x**3 +(5.994e-07)*y**3 + (2.528e-07)*y*x**2 + (4.815e-07)*x*y**2
            if a > 1024 or b > 1280 or a < 0 or b < 0:
                    continue
            a=int(a)
            b=int(b)
            img2[a,b] = px
        #DT.append(a)
        #DT2.append(b) #リストにデータを代入
        #img_white.item(a,b)=img.item(x,y) #修正後の座標に輝度値を代入

#画像に対して三次畳み込み保管を行う
cv2.imwrite('test.jpg',img2)
img3 = cv2.imread('test.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.show()

ksize=3

#中央値フィルタ
img_mask = cv2.medianBlur(img３,ksize)
plt.imshow(img_mask)
plt.show()



