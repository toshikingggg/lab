import noizu
import Flame

k = Flame.k
m = Flame.n

#Anを設定
#フレーム数の差分
def A(n):
    return n - k # kは最小のフレーム数
#Bnの設定
#界面位置の差分
def B(n):
    p = 1000/(m-k+1) #界面の検査領域/フレーム数
    return p*n - 200 # 計算の必要あり

def C(n): #AB
    p = 1000/(m-k+1) #界面の検査領域/フレーム数
    return (p*n-200) * (n-k)

def D(n): #AA
    return (n - k) ** 2

def sigma(func, frm, to): # シグマ関数
    result = 0
    for t in range(frm, to+1):
        result += func(t)
    #print(result)
    return result

print("sigma(A, k, m)の値:", sigma(A, k, m))
a = sigma(A, k, m)
print("sigma(B, k, m)の値:", sigma(B, k, m))
b = sigma(B, k, m)
print("sigma(C, k, m)の値:", sigma(C, k, m))
c = sigma(C, k, m)
print("sigma(D, k, m)の値:", sigma(D, k, m))
d = sigma(D, k, m)

V = ((m-k)*c - a*b)/((m-k)*d-a*a) #界面速度[pixel/flame]
print("界面速度[pixel/flame]:",V)

fps = 1000 #フレームレート[flame/sec]
"""
660pixel 100mm
"""
l = 0.011211 #1pixelあたりの長さ[mm/pixel]

u = fps*l*V

print("実世界の界面速度:",u)

pi = 3.14159265359
d = 2 #内径[mm]
Q = (pi*d*d*u*60)/(4*1000)

print("流量[ml/min]:",Q)