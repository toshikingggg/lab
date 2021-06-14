import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import pandas as pd
import time 
import numpy as np
import gc
from numba import njit

'''
Numba Performance Tips
https://numba.pydata.org/numba-doc/latest/user/performance-tips.html
'''

#ピンホール透過後の強度倍率計算用の関数
@njit
def Intensity_after_pinhole(x_i, slit_i):
    '''
    横(x軸側)の分解能は1nm,レーザーの強度分布は一定と仮定して縦方向の強度比率を計算
    '''
    #1nm = 1で記述
    #レーザーの縦の長さ→一旦4mmで計算
    beam_length = 4*(10**6)
    r_slit = slit_i / 2
    slit_beam_length = 2 * (np.sqrt((r_slit**2) - (x_i**2)))
    return slit_beam_length / beam_length

@njit
def calc_ratio_instance(instance, e_diameter):
    return instance/(e_diameter*2)

@njit
def light_receiving_t_d(gradient , intercept, x_k, y_k, distance,  slit, r_d, transmittance, change_step, sc, slit_t):
    
    #スリットの位置の定義
    #配列制作用
    const = r_d
    i = 0
    test_x = test_y = np.zeros(1, dtype=np.float64)
    
    # x1y1 = [[0 for _ in range(2)] for _ in range(const+1)]
    x1y1 = np.zeros((const+1, 2))
#     x1y1 = np.zeros((const*2 +1,2))
    # x2y2 = [[0 for _ in range(2)] for _ in range(const+1)]
    x2y2 = np.zeros((const+1, 2))
#     x2y2 = np.zeros((const*2 +1,2))
    #とりあえず外径の　1.5倍外
    
#     step = [i for i in range(-const, const +1)]
    
    #ここおかしくね？？→0.1ばいしてるから10倍分多くループ回す必要ある。。。。
    #for d in [tmp*0.1 for tmp in range(-const, const+1)]:
    slit_ti = slit_t
    #受光面の半径
    r_light = slit_ti + distance
    distance1 = distance
    distance2 = distance1 + slit_ti

    for d in np.arange(0, const):
        x1 = d
        y1 = -distance1
        x2 = d
        y2 = -distance2
        x1y1[i][0] = x1
        x1y1[i][1] = y1
        x2y2[i][0] = x2
        x2y2[i][1] = y2
        i += 1 
        # test_x.append(x1)
        # test_y.append(y1)


    # count_light = [0] * (const+1)
    count_light = np.zeros(const + 1, dtype=np.float64)
    
    # cnt_slit1 = 0
    # cnt_slit2 = 0
    
    count_x = np.zeros(1, dtype=np.float64)
    count_y = np.zeros(1, dtype=np.float64)
    #透過距離
    # distance_t = [i for i in range(const*2 + 1)]
    distance_t = np.arange(0, const*2 + 1)

    #test
    # length = len(gradient) - abs(change_step)
    # print("全体の長さ",len(gradient))
    # print("スイッチ1",change_step)
    # print("スイッチ2",length)
    # print("const",const)
    
    #ここのとり方を変える
    #zip使うと遅くなる？→普通のforループに書き直しても良いかも
    # cnt_raito = -1
    # change_ratio = 1
    for g, i, xk, yk, tm in zip(gradient, intercept, x_k, y_k, transmittance):
        if yk == 0:
            continue
        # cnt_raito += 1
        # if change_step < cnt_raito and cnt_raito < length:
        #     change_ratio = 1
        # else:
        #     change_ratio = 1
        # step_cnt = 0
        for j in np.arange(0, const): #-x,xで動かす
            # step_cnt += 1
            #スリットと,光線の交点
            a = 0
            b = x1y1[j][1]
            x = (i - b)/(a - g)
            #スリット幅(斜めver)
            s_d = slit/2
            #1つ目のスリットを超えられるかどうか
            if x < x1y1[j][0] + s_d and x1y1[j][0] - s_d < x:
                #2つめの条件に変更
                b = x2y2[j][1]
                x = (i - b)/(a - g)
                s_d = slit/2
                # cnt_slit1 += 1                
                #2つ目のスリットを超えられるかどうか
                if x < x2y2[j][0] + s_d and x2y2[j][0] - s_d < x:
                    # cnt_slit2 += 1
                    #ピンホールを透過する時のx座標を原点を中心として補正したx座標
                    pinhole_x = x - x2y2[j][0]
                    pinhole_intensity_ratio = Intensity_after_pinhole(pinhole_x, slit)
                    # tes.append(pinhole_intensity_ratio)
                    count_light[j] +=  pinhole_intensity_ratio * tm
#                     count_x.append(x)
#                     count_y.append(r_light) 

    # keep = []
    # keep = [i for i in count_light[::-1]]
    keep = count_light.copy()
    # np.delete(count_light, 0)
    count_light_all =  np.append(keep, count_light[1:][::-1])   
    # temp = keep + count_light                
    return count_light_all, distance_t, count_x, count_y, test_x, test_y



#透過角 を求めるための回転装置をもしたシミュレーション
#90度だけで良さそう
#全ての直線データをぶちこむ
#外周座標は最後の透過角を求めるために必要。
# import time
#回転中心が違うぽい
#TODO a可視化してみてみたほうがいいかも　test透過角のx:  -1574086.7167711235 -273693.0436330014　値がおかしそう？
@njit
def light_receiving_t_a(gradient ,intercept, x_k, y_k, distance, slit, center_x, center_y, transmittance, sc, slit_t):

    i = 0
    test_x = test_y = np.zeros(1, dtype=np.float64)
    # x1y1 = [[0 for _ in range(2)] for _ in range(90*sc)]
    # x2y2 = [[0 for _ in range(2)] for _ in range(90*sc)]
    x1y1 = np.zeros((90 * sc, 2))
    x2y2 = np.zeros((90 * sc, 2))
    # s_sita = [0 for _ in range(180*sc + 1)]
    s_sita = np.zeros(180*sc + 1)
    #スリットの厚み
    slit_ti = slit_t
    #受光面の半径
    r_light = slit_ti + distance
    for sita in [tmp*(1/sc) for tmp in np.arange(180*sc, 270*sc)]:
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
        # test_x.append(x1)
        # test_y.append(y1)
        
    #ある角度における光線の本数カウント用
    # count_light = [0] * (90 * sc + 1)
    count_light = np.zeros(90 * sc, dtype=np.float64)
    # count_x = [0] * (90 * sc)
    # count_y = [0] * (90 * sc)
#     t_angle = [0] * 900
    # t_angle = [0] * (90 * sc)
    t_angle = np.zeros(90 * sc, dtype=np.float64)
    
    count_x = np.zeros(1, dtype=np.float64)
    count_y = np.zeros(1, dtype=np.float64)
    
    for g, i, xk, yk, tm in zip(gradient, intercept, x_k, y_k, transmittance):
        if yk == 0:
            continue
        for j in np.arange(10*sc, 80*sc): #90度分の回転を表す
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
                    #ピンホールを透過する時のx座標を原点を中心として補正したx座標
                    pinhole_x = x - x2y2[j][0]
                    pinhole_intensity_ratio = Intensity_after_pinhole(pinhole_x, slit)
                    #光の本数をカウント
                    count_light[j] += tm * pinhole_intensity_ratio
                    # np.append(count_x, x)
                    y = -np.sqrt((r_light)**2 - (x)**2)
                    # np.append(count_y, y)
                    a = y - yk
                    b = xk - x
                    c = 1
                    d = 0
                    t_angle[j] = np.rad2deg(np.arctan(np.tan(abs(a*d - b*c)/abs(a*c + b*d))))
    
    return count_light, t_angle, count_x, count_y, test_x, test_y



for s_w in range(50000,50001,1000):
    for s_t in range(10, 11, 5):
        cnt = 0
        # scale = int(input("回転計測用の分解能を設定 ex)10→0.1°: "))
        scale = 100
        # slit_thickness = float(input("ピンホールの厚みを入力してください[mm]: "))
        slit_thickness = s_t
        slit_thickness *= 10**5
        # slit_width = float(input("ピンホールの幅を入力してください[mm]: "))
        slit_width = s_w
        # slit_width *= 10**6
        # f = float(input("レーザーの波長を入力してください[nm]: "))
        f = 632.8
        df_spe_sens = pd.read_csv('./dataset/Spectral_Sensitivity.csv')
        index = df_spe_sens.query('lam == {}'.format(f)).index.tolist()
        #PTによって変化しそうなので一旦理想的な１とする
        # Spectral_Sensitivity = df_spe_sens['C'][index]
        Spectral_Sensitivity = 0.95
        #[nW/m^2]
        lx_to_nW = 1.46 * (10**6) / Spectral_Sensitivity
        for step in range(100000, 100001, 5000):
            for step_i in range(50000, 50001, 1):
                #スリットと細管の距離
                for step_bt_slit_and_tube in range(1000000,1000001, 100000):
                    #1mm → 100000  0.2mm → 20000 0.05mm → 5000
                    r = step
                    r_i = step_i
                    r_i = round(r_i)
                    #複数のCSVファイルを順次読み込んでグラフを表示していく
                    df = pd.read_csv('./new_dataset/dataset_r_i_{0}_r_{1}.csv'.format(r_i, r))
                    new_x_a_list_t = df['x'] 
                    new_y_a_list_t = df['y']
                    gradient_list_t = df['gradient']
                    intercept_list_t = df['intercept']
                    center_x_list_t = df['change_x']
                    center_y_list_t = df['change_y']
                    step_change_e_t = df['step_change_end']
                    transmittance_slist_t = df['transmittance_s']
                    transmittance_plist_t = df['transmittance_p']
                    transmittance_list_t = transmittance_slist_t + transmittance_plist_t
                    new_x_a_list = new_x_a_list_t.values
                    new_y_a_list= new_y_a_list_t.values
                    gradient_list = gradient_list_t.values
                    intercept_list = intercept_list_t.values
                    center_x_list = center_x_list_t.values
                    center_y_list = center_y_list_t.values
                    step_change_e = step_change_e_t.values
                    transmittance_slist = transmittance_slist_t.values
                    transmittance_plist = transmittance_plist_t.values
                    transmittance_list = transmittance_list_t.values
                    if center_x_list[0] == 0 and center_y_list[0] == 0:
                        print('回転中心を定義できないため「dataset_r_i_{0}_r_{1}.csv」を飛ばしました'.format(r_i, r))
                        print('--------------------------------------------------------------')
                        continue
                    #if step == 34000 and step_i < 5000:
                        #continue
                    flag = False
            #     for threshold in [tmp*0.1 for tmp in range(1, 10)]:
                    r = step
                    r_i = step_i
                    r_i = round(r_i)
                #円管とスリットの距離

                    #TODO ピンホールで計測したいのでここ変更
                    #細管とピンホールの距離
                    h = step_bt_slit_and_tube
                    d = r + h
                    #スリット幅
                    s = slit_width
                #         t1 = time.time() 
                    cnt += 1
                    print('「dataset_r_i_{0}_r_{1}.csv」を解析中'.format(r_i, r))
                    print('d={0} r_i={1} r={2} slit_thickness={3} slit_width={4} bt_slit_and_tube={5}'.format(d/1000000, r_i/1000000, r/1000000, slit_thickness/1000000, slit_width/1000000, step_bt_slit_and_tube/1000000))

                    light_num_a, toka_angle,light_x1,light_y1,debug_x1,debug_y1 = light_receiving_t_a(gradient_list ,intercept_list, new_x_a_list, new_y_a_list, d, s , center_x_list[0], center_y_list[0], transmittance_list, scale, slit_thickness)

                    #TODO light_numに対して最小検出Wを定義する必要あり 1W →　10^9されてることに注意

                    # light_num_a.pop(0)
                    # print("DONE",np.sum(light_num_a))
                    # plt.figure(0)
                    # plt.bar(toka_angle, light_num_a, width=1.0)
                    # plt.show()
        #             exit()
                    light_num_d, distance,light_x,light_y,debug_x2,debug_y2 = light_receiving_t_d(gradient_list ,intercept_list, new_x_a_list, new_y_a_list, d , s, r, transmittance_list, step_change_e[0], scale, slit_thickness)
                    # print("test透過角のx: ",min(light_x1),max(light_x1))
                    print("DONE")
                    # exit()
                    # print("debug：強度",min(light_x),max(light_x))
                    print("debug,強度: ",sum(light_num_d),sum(light_num_a))
                    # print("debug,pinhole: ",(min(debug_3),max(debug_3)))
                    # print("tes",len(light_num_d),len(distance))
                    #参考値
                    print("透過角")
                    print(max(light_num_a))
                    print("透過距離")
                    print(max(light_num_d))

                    print("比較a",max(light_num_a) / (((((1.7 * (10**6)) // 2) * (10**(-9)))**2) * np.pi), '閾値', lx_to_nW)
                    print("比較d",max(light_num_d) / (((((1.7 * (10**6)) // 2) * (10**(-9)))**2) * np.pi), '閾値', lx_to_nW)
                    # exit()
                    #受光部分の最小検出強度[nW] λ=555の時のlxーW変換値→比視感度は実験値参考。1.46[W/m^2]→1[lx]
                    #need to change
                    
                    for i in np.arange(len(light_num_a)):
                        # if light_num_a[i] / (0.46 * 0.32) <= lx_to_nW.iloc[-1]:
                        if light_num_a[i] / (((((1.7 * (10**6)) // 2) * (10**(-9)))**2) * np.pi) <= lx_to_nW:
                            light_num_a[i] = 0
                    
                    for i in np.arange(len(light_num_d)):
                        # if light_num_d[i] <= lx_to_nW.iloc[-1]:
                        if light_num_d[i] / (((((1.7 * (10**6)) // 2) * (10**(-9)))**2) * np.pi) <= lx_to_nW:
                            light_num_d[i] = 0    
                    
                    if max(light_num_a) == 0 or max(light_num_d) == 0:
                        print("閾値を超えていません", "外径",2*step/1000000,"[mm]","内径",2*step_i/1000000,"[mm]","ピンホールと細管の距離",step_bt_slit_and_tube/1000000,"[mm]")
                        del light_num_a
                        del toka_angle
                        del light_x1
                        del light_y1
                        del debug_x1
                        del debug_y1
                        del light_num_d
                        del distance
                        del light_x
                        del light_y
                        del debug_x2
                        del debug_y2
                        gc.collect()
                        flag = True
                        
                    if flag == False:

                        
                        # print("tes",len(light_num_d),len(distance))
                        fig = plt.figure(figsize=(12,6))
                        #         fig.subplots_adjust(wspace=0.5)
                        ax1 = fig.add_subplot(1, 2, 1)
                        ax2 = fig.add_subplot(1, 2, 2)
                        ax1.set_xlim([0,90])
                        ax1.bar(toka_angle, light_num_a, width=1.0)
                        ax2.bar(distance, light_num_d, width=1.0)
                        ax1.set_title('Inner_diameter = {0}mm External_diameter = {1}mm'.format(r_i*2/100000, r*2/100000))
                        ax2.set_title('Inner_diameter = {0}mm External_diameter = {1}mm'.format(r_i*2/100000, r*2/100000))
                        ax1.set_xlabel('Transmission angle[°]')
                        ax2.set_xlabel('Distance[μm]')
                        ax1.set_ylabel('Voltage')
                        ax2.set_ylabel('Voltage')
                        ax1.tick_params(direction = "out")
                        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f'{x//1000}'))
                        ax2.tick_params(direction = "out")
                        ax1.set_yticklabels([])
                        ax2.set_yticklabels([])
                        #plt.show()
                        # fig.savefig('./data_csv_test_previous/figure_d1_{0}_d2_{1}_r_i_{2}_r_{3}.png'.format(d1, d2, r_i, r))
                        fig.savefig('./new_figure/figure_d_{0}_r_i_{1}_r_{2}_slit_thickness_{3}_slit_width_{4}_bt_slit_and_tube_{5}.png'.format(d, r_i, r, slit_thickness, slit_width, step_bt_slit_and_tube))
                        plt.close(fig)
                        # exit()
                        
                        #計算用
                        inner_diameter_distance = []
                        inner_diameter_angle = []
                        expected_innner_diameter_1 = []
                        expected_innner_diameter_2 = []
                        num_1 = len(light_num_d)
                        inner_diameter_distance = [0]*num_1
                        expected_innner_diameter_1 = [0]*num_1
                        num_2 = len(light_num_a)
                        inner_diameter_angle = [0]*num_2
                        expected_innner_diameter_2 = [0]*num_2

                        tmp = 0
                        tmp2 = 0
                        for i in range(len(light_num_a)):   
                            if tmp < light_num_a[i]:
                                tmp = light_num_a[i]
                                tmp2 = toka_angle[i]

                        # print("光線数",tmp,"透過角",tmp2)

                        tmp3 = 0
                        tmp4 = 0

                        for i in range(int(len(light_num_d)/2)):   
                            if tmp3 < light_num_d[i]:
                                tmp3 = light_num_d[i]
                                tmp4 = distance[i]

                        max_value = max(light_num_d)
                        # max_index = light_num_d.index(max_value)
                        max_index = np.where(light_num_d == max_value)
                        tmp3 = distance[max_index]
                        # print(tmp)
                        # a_ratio_instance = calc_ratio_instance(tmp, r)
                        # d_ratio_instance = calc_ratio_instance(tmp3, r)
                        # print("強度比")
                        # print("透過角",a_ratio_instance * 1000,"μW")
                        # print("透過距離",d_ratio_instance * 1000,"μW")
                        # print("光線数",tmp,"透過距離",(distance[-1]-tmp3*2)/10000,"mm")
                        print("-----------------------------------------------------")
                        D = 2*r/1000000
                        n = 1.49/1.000292
                        #透過角からの計算
                        sita_ans = np.deg2rad(tmp2)
                        d_ans_3 = np.sqrt((pow(D,2)*pow(np.sin(sita_ans/2),2))/(pow(n,2)-2*n*np.cos(sita_ans/2)+1))
                        print("透過角からの計算")
                        print(d_ans_3)
                        print("-----------------------------------------------------")

                        #透過距離からの計算
                        a_toka = (distance[-1]-tmp3*2)/1000000

                        d_ans_1 = (-pow(a_toka,3)-np.sqrt(pow(a_toka,6)+pow(a_toka,2)*pow(D,2)*(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))))/(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))
                        d_ans_2 = (-pow(a_toka,3)+np.sqrt(pow(a_toka,6)+pow(a_toka,2)*pow(D,2)*(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))))/(pow(n,2)*(pow(D,2)-pow(a_toka,2))-pow(a_toka,2))
                        print("透過距離からの計算")
                        print("内径1:",d_ans_1,"内径2:",d_ans_2)
                        d_ans = float('inf')
                        for d1_i in d_ans_1:
                            if abs(r_i*2 - d_ans*1000000) > abs(r_i*2 - d1_i*1000000):
                                d_ans = d1_i
                        for d2_i in d_ans_2:
                            if abs(r_i*2 - d_ans*1000000) > abs(r_i*2 - d2_i*1000000):
                                d_ans = d2_i    
                            
                        print(d_ans)
                        print("-----------------------------------------------------")
                        # 期待する内径
                        print("期待する内径")
                        print(r_i*2/1000000,"mm")
                        inner_diameter_distance[0] = d_ans
                        inner_diameter_angle[0] = d_ans_3
                        expected_innner_diameter_1[0] = r_i*2/1000000
                        expected_innner_diameter_2[0] = r_i*2/1000000

                        df = pd.DataFrame({
                            'through_strength_distance':light_num_d,
                            'distance':distance,
                            'inner_diameter_distance':inner_diameter_distance,
                            'expected_innner_diameter':expected_innner_diameter_1
                        })

                        df.to_csv('./new_data/distance_d_{0}_r_i_{1}_r_{2}_slit_thickness_{3}_slit_width_{4}_bt_slit_and_tube_{5}.csv'.format(d, r_i, r, slit_thickness, slit_width, step_bt_slit_and_tube), index=False)

                        df2 = pd.DataFrame({
                            'through_strength_angle':light_num_a,
                            'through_angle':toka_angle,
                            'inner_diameter_angle':inner_diameter_angle,
                            'expected_innner_diameter':expected_innner_diameter_2
                        })

                        df2.to_csv('./new_data/angle_d1_{0}_r_i_{1}_r_{2}_slit_thickness_{3}_slit_width_{4}_bt_slit_and_tube_{5}.csv'.format(d, r_i, r, slit_thickness, slit_width, step_bt_slit_and_tube), index=False)

                    if flag:
                        del df
                        del new_x_a_list
                        del new_y_a_list
                        del gradient_list
                        del intercept_list
                        del center_x_list
                        del center_y_list
                        del transmittance_slist
                        del transmittance_plist
                        del transmittance_list
                        gc.collect()
                    else:
                        del df
                        del new_x_a_list
                        del new_y_a_list
                        del gradient_list
                        del intercept_list
                        del center_x_list
                        del center_y_list
                        del transmittance_slist
                        del transmittance_plist
                        del transmittance_list
                        del light_num_a
                        del toka_angle
                        del light_x1
                        del light_y1
                        del debug_x1
                        del debug_y1
                        del light_num_d
                        del distance
                        del light_x
                        del light_y
                        del debug_x2
                        del debug_y2
                        del inner_diameter_distance
                        del inner_diameter_angle
                        del expected_innner_diameter_1
                        del expected_innner_diameter_2
                        del df2
                        gc.collect()