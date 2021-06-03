import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import time 
import numpy as np
import gc
from numba import jit

def calc_ratio_instance(instance, e_diameter):
    return instance/(e_diameter*2)

def light_receiving_t_d(gradient , intercept, x_k, y_k, distance1, distance2, slit1, slit2, r_d, transmittance, change_step):
    
    #スリットの位置の定義
    #配列制作用
    const = r_d
    i = 0
    test_x = []
    test_y = []
    
    x1y1 = [[0 for _ in range(2)] for _ in range(const+1)]
#     x1y1 = np.zeros((const*2 +1,2))
    x2y2 = [[0 for _ in range(2)] for _ in range(const+1)]
#     x2y2 = np.zeros((const*2 +1,2))
    #とりあえず外径の　1.5倍外
    
#     step = [i for i in range(-const, const +1)]
    
    #ここおかしくね？？→0.1ばいしてるから10倍分多くループ回す必要ある。。。。
    #for d in [tmp*0.1 for tmp in range(-const, const+1)]:
    for d in range(0, const):
        x1 = d
        y1 = -distance1
        x2 = d
        y2 = -distance2
        x1y1[i][0] = x1
        x1y1[i][1] = y1
        x2y2[i][0] = x2
        x2y2[i][1] = y2
        i += 1 
        test_x.append(x1)
        test_y.append(y1)


    count_light = [0] * (const+1)
    
    cnt_slit1 = 0
    cnt_slit2 = 0
    
    count_x = []
    count_y = []
    step_cnt = 0
    step = 1
    #透過距離
    distance_t = [i for i in range(const*2 + 1)]

    #test
    length = len(gradient) - abs(change_step)
    print("全体の長さ",len(gradient))
    print("スイッチ1",change_step)
    print("スイッチ2",length)
    print("const",const)
#     distance_t = np.arange(const * 2 + 1)
    #ここのとり方を変える
    #zip使うと遅くなる？→普通のforループに書き直しても良いかも
    cnt_raito = -1
    change_ratio = 1
    for g, i, xk, yk, tm in zip(gradient, intercept, x_k, y_k, transmittance):
        if yk == 0:
            continue
        #　ここで光の倍率を変える必要がありそう
        cnt_raito += 1
        if change_step < cnt_raito and cnt_raito < length:
            change_ratio = 0.1
        else:
            change_ratio = 1
        # step_cnt = 0
        for j in range(0, const): #-x,xで動かす
            # step_cnt += 1
            #スリットと,光線の交点
            a = 0
            b = x1y1[j][1]
            x = (i - b)/(a - g)
            #スリット幅(斜めver)
            s_d = slit1/2
            #1つ目のスリットを超えられるかどうか
            ##この条件がおかしそう
            if x < x1y1[j][0] + s_d and x1y1[j][0] - s_d < x:
                #2つめの条件に変更
                #ここの計算間違っている説あり（要確認)
                b = x2y2[j][1]
                x = (i - b)/(a - g)
                s_d = slit2/2
                cnt_slit1 += 1
                #2つ目のスリットを超えられるかどうか
                if x < x2y2[j][0] + s_d and x2y2[j][0] - s_d < x:
                    cnt_slit2 += 1
                    #光の本数をカウント
                    # print("j",j)
                    # if change_step < j:
                    #     count_light[j] += tm
                    #     print("条件分岐確認")
                    # else:
                    #     count_light[j] += tm*0.01
                    count_light[j] += tm*change_ratio
                    count_x.append(x)
                    y = g*x + i
                    count_y.append(y) 
              
    keep = []
    keep = [i for i in count_light[::-1]]
    count_light.pop(0)
    count_light =  keep + count_light      
    # temp = keep + count_light                
    return count_light, distance_t, count_x, count_y, test_x, test_y



#透過角 を求めるための回転装置をもしたシミュレーション
#90度だけで良さそう
#全ての直線データをぶちこむ
#外周座標は最後の透過角を求めるために必要。
# import time
#回転中心が違うぽい
@jit
def light_receiving_t_a(gradient ,intercept, x_k, y_k, distance, slit, center_x, center_y, transmittance, sc):

    i = 0
    test_x = []
    test_y = [] 
    x1y1 = [[0 for _ in range(2)] for _ in range(90*sc)]
    x2y2 = [[0 for _ in range(2)] for _ in range(90*sc)]
    s_sita = [0 for _ in range(180*sc + 1)]
    #スリットの厚み
    slit_ti = 1.58 * (10**6)
    for sita in [tmp*(1/sc) for tmp in range(180*sc, 270*sc)]:
        x1 = distance*np.cos(np.deg2rad(sita)) + center_x
        y1 = distance*np.sin(np.deg2rad(sita)) + center_y
        x2 = (distance+slit_ti)*np.cos(np.deg2rad(sita)) + center_x
        y2 = (distance+slit_ti)*np.sin(np.deg2rad(sita)) + center_y
        similar_sita = abs(np.arctan(np.tan(abs(x1)/abs(y1))))
        x1y1[i][0] = x1
        x1y1[i][1] = y1
        x2y2[i][0] = x2
        x2y2[i][1] = y2

        s_sita[i] = similar_sita
        i += 1
        test_x.append(x1)
        test_y.append(y1)
        
    #ある角度における光線の本数カウント用
    count_light = [0] * (90 * sc + 1)
    # count_x = [0] * (90 * sc)
    # count_y = [0] * (90 * sc)
#     t_angle = [0] * 900
    t_angle = [0] * (90 * sc)
    
    count_x = []
    count_y = []
    
    for g, i, xk, yk, tm in zip(gradient, intercept, x_k, y_k, transmittance):
        if yk == 0:
            continue
        for j in range(10*sc, 80*sc): #90度分の回転を表す
            #スリットの傾きと,光線の交点
            a = -np.tan(s_sita[j])
            b = x1y1[j][1] - a*x1y1[j][0]
            x = (i - b)/(a - g)
            #スリット幅(斜めver)
            s_d = slit*np.cos(abs(s_sita[j]))/2
            #1つ目のスリットを超えられるかどうか
            if x < x1y1[j][0] + s_d and x1y1[j][0] - s_d < x:
                #2つめの条件に変更
                b = x2y2[j][1] - a*x2y2[j][0]
                x = (i - b)/(a - g)
                s_d = slit*np.cos(abs(s_sita[j]))/2
                if x < x2y2[j][0] + s_d and x2y2[j][0] - s_d < x:
                    #光の本数をカウント
                    count_light[j] += tm
                    # count_x.append(x)
                    y = g*x + i
                    # count_y.append(y)
                    a = y - yk
                    b = xk - x
                    c = 1
                    d = 0
                    t_angle[j] = np.rad2deg(np.arctan(np.tan(abs(a*d - b*c)/abs(a*c + b*d))))
    
    return count_light, t_angle, count_x, count_y, test_x, test_y

cnt = 0
scale = int(input("計測用の分解能を設定 ex)10→0.1mm"))
for step in range(10000, 10001):
    for step_i in range(5000, 5001, 1):
        #if step == 34000 and step_i < 5000:
            #continue
            
#     for threshold in [tmp*0.1 for tmp in range(1, 10)]:
        r = step
        r_i = step_i
        r_i = round(r_i)
    #1mm → 100000  0.2mm → 20000 0.05mm → 5000
    #円管とスリットの距離

        #TODO ピンホールで計測したいのでここ変更
        #細管とピンホールの
        h = 100000
        d = r + h
        #スリット幅
        s = 5000
    #         t1 = time.time() 
        #複数のCSVファイルを順次読み込んでグラフを表示していく
        df = pd.read_csv('./dataset/dataset_r_i_{0}_r_{1}.csv'.format(r_i, r))
        new_x_a_list = df['x'] 
        new_y_a_list = df['y']
        gradient_list = df['gradient']
        intercept_list = df['intercept']
        center_x_list = df['change_x']
        center_y_list = df['change_y']
        step_change_e = df['step_change_end']
        transmittance_slist = df['transmittance_s']
        transmittance_plist = df['transmittance_p']
        transmittance_list = [s+p for s,p in zip(transmittance_slist ,transmittance_plist)]
        cnt += 1
        if center_x_list[0] == 0 and center_y_list[0] == 0:
            print('回転中心を定義できないため「dataset_r_i_{0}_r_{1}.csv」を飛ばしました'.format(r_i, r))
            print('--------------------------------------------------------------')
            continue
        print('「dataset_r_i_{0}_r_{1}.csv」を解析中'.format(r_i, r))

        light_num_a, toka_angle,light_x1,light_y1,debug_x1,debug_y1 = light_receiving_t_a(gradient_list ,intercept_list, new_x_a_list, new_y_a_list, d, s , center_x_list[0], center_y_list[0], transmittance_list, scale)

        #TODO light_numに対して最小検出Wを定義する必要あり 1W →　10^9されてることに注意

        light_num_a.pop(0)

        # light_num_d, distance,light_x,light_y,debug_x2,debug_y2 = light_receiving_t_d(gradient_list ,intercept_list, new_x_a_list, new_y_a_list, d1, d2,s1 ,s2, r, transmittance_list, step_change_e[0], scale)
    
        # print("tes",len(light_num_d),len(distance))
        fig = plt.figure(figsize=(12,6))
        #         fig.subplots_adjust(wspace=0.5)
        ax1 = fig.add_subplot(1, 2, 1)
        # ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_xlim([0,90])
        ax1.bar(toka_angle, light_num_a, width=1.0)
        # ax2.bar(distance, light_num_d, width=1.0)
        ax1.set_title('Inner_diameter = {0}mm External_diameter = {1}mm'.format(r_i*2/100000, r*2/100000))
        # ax2.set_title('Inner_diameter = {0}mm External_diameter = {1}mm'.format(r_i*2/100000, r*2/100000))
        ax1.set_xlabel('Transmission angle')
        # ax2.set_xlabel('Distance')
        ax1.set_yticklabels([])
        #         ax2.set_yticklabels([])
        ax1.tick_params(length=0)
        #         ax2.tick_params(length=0)
        #plt.show()
        # fig.savefig('./data_csv_test_previous/figure_d1_{0}_d2_{1}_r_i_{2}_r_{3}.png'.format(d1, d2, r_i, r))
        fig.savefig('./figure/figure_d_{0}_r_i_{1}_r_{2}.png'.format(d, r_i, r))
        plt.close(fig)


        # #計算用
        # inner_diameter_distance = []
        # inner_diameter_angle = []
        # expected_innner_diameter_1 = []
        # expected_innner_diameter_2 = []
        # num_1 = len(light_num_d)
        # inner_diameter_distance = [0]*num_1
        # expected_innner_diameter_1 = [0]*num_1
        # num_2 = len(light_num_a)
        # inner_diameter_angle = [0]*num_2
        # expected_innner_diameter_2 = [0]*num_2

        # tmp = 0
        # tmp2 = 0
        # for i in range(len(light_num_a)):   
        #     if tmp < light_num_a[i]:
        #         tmp = light_num_a[i]
        #         tmp2 = toka_angle[i]

        # # print("光線数",tmp,"透過角",tmp2)

        # tmp3 = 0
        # tmp4 = 0

        # for i in range(int(len(light_num_d)/2)):   
        #     if tmp3 < light_num_d[i]:
        #         tmp3 = light_num_d[i]
        #         tmp4 = distance[i]

        # max_value = max(light_num_d)
        # max_index = light_num_d.index(max_value)
        # tmp3 = distance[max_index]
        # # print(tmp)
        # a_ratio_instance = calc_ratio_instance(tmp, r)
        # d_ratio_instance = calc_ratio_instance(tmp3, r)
        # print("強度比")
        # print("透過角",a_ratio_instance * 1000,"μW")
        # print("透過距離",d_ratio_instance * 1000,"μW")
        # # print("光線数",tmp,"透過距離",(distance[-1]-tmp3*2)/10000,"mm")
        # # print("-----------------------------------------------------")
        # #透過距離からの計算
        # a_toka = (distance[-1]-tmp3*2)/10000
        # D = 2*r/10000
        # n = 1.49/1.000292

        # d_ans_1 = (-pow(a_toka,3)-np.sqrt(pow(a_toka,6)+pow(a_toka,2)*pow(D,2)*(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))))/(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))
        # d_ans_2 = (-pow(a_toka,3)+np.sqrt(pow(a_toka,6)+pow(a_toka,2)*pow(D,2)*(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))))/(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))
        # # print("透過距離からの計算")
        # # print("内径1:",d_ans_1,"内径2:",d_ans_2)
        # # print("-----------------------------------------------------")

        # if r_i*2 - d_ans_1*10000 < r_i*2 - d_ans_2*10000:
        #     d_ans = d_ans_1
        # else:
        #     d_ans = d_ans_2

        # # print(d_ans)
        # #透過角からの計算
        # sita_ans = np.deg2rad(tmp2)
        # d_ans_3 = np.sqrt((pow(D,2)*pow(np.sin(sita_ans/2),2))/(pow(n,2)-2*n*np.cos(sita_ans/2)+1))
        # # print("透過角からの計算")
        # # print(d_ans_3)
        # # print("-----------------------------------------------------")
        # #期待する内径
        # # print("期待する内径")
        # # print(r_i*2/10000,"mm")

        # inner_diameter_distance[0] = d_ans
        # inner_diameter_angle[0] = d_ans_3
        # expected_innner_diameter_1[0] = r_i*2/10000
        # expected_innner_diameter_2[0] = r_i*2/10000

        # df = pd.DataFrame({
        #     'through_strength_distance':light_num_d,
        #     'distance':distance,
        #     'inner_diameter_distance':inner_diameter_distance,
        #     'expected_innner_diameter':expected_innner_diameter_1
        # })

        # df.to_csv('./data/distance_d1_{0}_d2_{1}_r_i_{2}_r_{3}.csv'.format(d1, d2, r_i, r), index=False)

        # df2 = pd.DataFrame({
        #     'through_strength_angle':light_num_a,
        #     'through_angle':toka_angle,
        #     'inner_diameter_angle':inner_diameter_angle,
        #     'expected_innner_diameter':expected_innner_diameter_2
        # })

        # df2.to_csv('./data/angle_d1_{0}_d2_{1}_r_i_{2}_r_{3}.csv'.format(d1, d2, r_i, r), index=False)


        # del df
        # del new_x_a_list
        # del new_y_a_list
        # del gradient_list
        # del intercept_list
        # del center_x_list
        # del center_y_list
        # del transmittance_slist
        # del transmittance_plist
        # del transmittance_list
        # del light_num_a
        # del toka_angle
        # del light_x1
        # del light_y1
        # del debug_x1
        # del debug_y1
        # del light_num_d
        # del distance
        # del light_x
        # del light_y
        # del debug_x2
        # del debug_y2
        # del inner_diameter_distance
        # del inner_diameter_angle
        # del expected_innner_diameter_1
        # del expected_innner_diameter_2
        # del df2
        # gc.collect()