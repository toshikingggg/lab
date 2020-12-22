# coding: utf-8

from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d # 3D表示に使う
from scipy import linalg # linalg.solve(A, b) 　Ax = bの解を求める関数

"""
位置校正用の格子板の交点を既知点として抽出
そして補正式を３次の近似多項式として係数を最小二乗法を用いて求め、濃度値は３次畳み込み内挿法で内挿
その後得られた補正式の結果を使い画像を補正
"""

"""img_gray = np.array(Image.open('/Volumes/Transcend/卒業研究/program/DirectMapping/test/図1.jpg').convert('L'))

print(img_gray)"""

img = cv2.imread('/Volumes/Transcend/卒業研究/program/DirectMapping/test/図1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,13,3,0.04)
#dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# 中心見つける
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# コーナーをサブピクセル精度で検出、また基準を設ける
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# 描画
res = np.hstack((centroids,corners))
res = np.int0(res)

#img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

"""#cv2.imwrite('subpixel5.png',img)
plt.imshow(img)
plt.show()
"""
#print(corners)
#格子点の座標を抽出

newcorners = np.delete(corners, 0, 0)
print(newcorners) 

x,y = np.split(newcorners, 2, 1) #x,y軸それぞれについて分ける
print(x)
print(y)
print(len(newcorners))

plt.imshow(img)
plt.show()

"""
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst',img)
cv2.waitKey(0) 
cv2.destroyAllWindows()
"""
"""
#サブピクセル精度で検出した座標を与えて、多項式の係数を最小二乗法で求める。
#x,y = np.split(newcorners, 2, 1) #x,y軸それぞれについて分ける
x = newcorners[:,0]
y = newcorners[:,1]
f = 1 + x + y + x**2 + y**2 + x*y + x**3 + y**3 + y*x**2 + x*y**2

Xtil = np.c_[np.ones(newcorners.shape[0]), newcorners] # Xの行列の左端に[1,1,1,...,1]^Tを加える。
A = np.dot(Xtil.T, Xtil) # 標準形A,bに当てはめる。
b = np.dot(Xtil.T, f)
w = linalg.solve(A, b) # (8)式をwについて解く。

xmesh, ymesh = np.meshgrid(np.linspace(0, 1000, 20),
                            np.linspace(0,1000, 20))
zmesh = (w[0] + w[1] * xmesh.ravel() +
        w[2] * ymesh.ravel() ).reshape(xmesh.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, f, color='k')
ax.plot_wireframe(xmesh, ymesh, zmesh, color='r')
plt.show()
"""
#print(w)
"""
def setDT(N):#データ設定
    DT = newcorners
    #for i in range(N+1): DT.append(math.sin(math.pi*i/3))
    return DT
def cubicF(t,alfa):#３次畳み込み関数による近似
    t1=abs(t); t2=t1*t1; t3=t2*t1
    if t1 >= 2: return 0
    if t1 >= 1: return alfa*t3 - 5*alfa*t2 + 8*alfa*t1 - 4*alfa
    return (alfa + 2)*t3 - (alfa + 3)*t2 + 1
def cubicInterpo(DT,alfa):#三次畳み込み関数による補間
    N=len(DT); t=0;V=0;R=[]
    for i in range(N):
        for j in range(10):
            t=0.1*j;   V = DT[i  ]*cubicF(t  ,alfa)
            if i>=1:   V+= DT[i-1]*cubicF(t+1,alfa)
            if i<=N-3: V+= DT[i+2]*cubicF(t-2,alfa)
            if i<=N-2: V+= DT[i+1]*cubicF(t-1,alfa)
            #print(t+i,",",V)
            R.append([V])
    return R

DT=setDT(a) 
alfa=-0.5 
S=cubicInterpo(DT,alfa)
for i in range(len(S)):
     print(S[i][0])
     img2=cv2.drawMarker(img, (S[i][0][0], S[i][0][1]), (255, 0, 0))

cv2.imshow('img',img2)
cv2.waitKey(0)
"""
""" 
def func(x,y,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9):
    return a0 + a1*x + a2*y + a3*x**2 + a4*y**2 + a5*x*y + a6*x**3 + a7*y**3 + a8*y*x**2 + a9*x*y**2
    
res = optimize.curve_fit(func, X, Y) # 

print(res)
"""
#再配列、出力画像の各画素について入力画像の座標系での位置を計算する（逆変換）三次畳み込み内挿