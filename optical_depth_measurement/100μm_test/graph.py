#内径から一番合ってる外径を求める
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import time 
import numpy as np
import seaborn as sns
import gc
import matplotlib.cm as cm

inner_diameter_distance = []
inner_diameter_angle = []
scatter_a = []
scatter_d = []
expectation = []
ratio = []
temp_r_i = []
temp_r = []
i = 0
j = 0
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3"]
colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
for step_bt_slit_and_tube in range(100000, 1000001, 100000):
    # for step_i in range(50000, 50001, 1):
    step_i = 50000
    inner_diameter_distance = []
    inner_diameter_angle = []
    scatter_a = []
    scatter_d = []
    expectation = []
    ratio = []
    temp_r_i = []
    temp_r = []
    outer_radius_change = []
    change_hole_to_tube = []
    i = 0
    for step in range(75000, 150001, 5000):
        #need to change
        # step_bt_slit_and_tube = 100000
        slit_thickness = 1000000
        slit_width = 50000
        r = step
        r_i = step_i
        r_i = round(r_i)
        h = step_bt_slit_and_tube
        d = r + h
        #それぞれにわける
        try:
            df2 = pd.read_csv('./new_data/distance_d_{0}_r_i_{1}_r_{2}_slit_thickness_{3}_slit_width_{4}_bt_slit_and_tube_{5}.csv'.format(d, r_i, r, slit_thickness, slit_width, step_bt_slit_and_tube))
            # df1 = pd.read_csv('./new_data/angle_d1_{0}_r_i_{1}_r_{2}_slit_thickness_{3}_slit_width_{4}_bt_slit_and_tube_{5}.csv'.format(d, r_i, r, slit_thickness, slit_width, step_bt_slit_and_tube))
        except:
            continue
        #横軸 求めた内径 縦軸 バラツキ？expectedとの差
        # inner_diameter_distance.append(df2['inner_diameter_distance'][0])
        # inner_diameter_angle.append(df1['inner_diameter_angle'][0])
        # expectation.append(df1['expected_innner_diameter'][0])
        expectation.append(df2['expected_innner_diameter'][0])
        # scatter_a.append(abs((df1['inner_diameter_angle'][0] - expectation[i])/expectation[i]))
        scatter_d.append(abs((df2['inner_diameter_distance'][0] - expectation[i])/expectation[i]))
        outer_radius_change.append(r * 2 / 1000000)
        change_hole_to_tube.append(h / 1000000)

        #比率
        ratio.append((r-r_i)/r_i)
        #内径と外径
        temp_r_i.append(r_i)
        temp_r.append(r)

        i += 1
        # del df1
        del df2
        gc.collect()


        #誤差が一番小さい外径と内径
        # inner_diameter_a = temp_r_i[scatter_a.index(min(scatter_a))]
        inner_diameter_d = temp_r_i[scatter_d.index(min(scatter_d))]
        # external_diameter_a = temp_r[scatter_a.index(min(scatter_a))]
        external_diameter_d = temp_r[scatter_d.index(min(scatter_d))]
        # print('-------------------------------')
        # print('内径r={}mmに対して'.format(r_i*2/10000))
        # print("理想的な条件(誤差が一番小さくなる)外径")
        # print("透過角　","外径:",external_diameter_a*2/1000000, "mm")
        # print("肉厚:内径=",(external_diameter_a-inner_diameter_a)/inner_diameter_d,":",1)
        # print("誤差！",min(scatter_a)*100,"%")
        # print("透過距離　 ","外径:",external_diameter_d*2/1000000, "mm")
        # print("肉厚:内径=",(external_diameter_d-inner_diameter_d)/inner_diameter_d,":",1)
        # print("誤差！",min(scatter_d)*100,"%")
        # print('-------------------------------')
    #outer_radius_change

    # plt.plot(outer_radius_change, scatter_a, marker=markers[j], color='black', markeredgecolor='black', markerfacecolor='none',linewidth = 0.5, linestyle='solid', label='d = '+str(step_bt_slit_and_tube/1000000))
    plt.plot(outer_radius_change, scatter_d, marker=markers[j], color='black', markeredgecolor='black', markerfacecolor='none',linewidth = 0.5, linestyle='solid', label='d = '+str(step_bt_slit_and_tube/1000000))
    
    # plt.plot(change_hole_to_tube, scatter_d, marker="o", color='black', markeredgecolor='blue', markerfacecolor='none')
    plt.legend()
    plt.ylim([0,1.0])
    plt.savefig('./res/distance_r_i_{0}_btw_{1}.png'.format(r_i,step_bt_slit_and_tube))
    j += 1
    # 折れ線グラフを出力
    # fig = plt.figure(figsize=(12,6))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax1.plot(outer_radius_change, scatter_a, marker="o")
    # ax2.plot(outer_radius_change, scatter_d, marker="o")
    # fig.savefig('./tes/r_i_{}.png'.format(r_i))
    # カラーマップ
    # cm = plt.cm.get_cmap('seismic')
    # fig = plt.figure(figsize=(12,6))
    # #         fig.subplots_adjust(wspace=0.5)
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # #割合で計算しようね
    # #a_sc = ax1.scatter(expectation, scatter_a, s = 10, c = ratio, vmin=0, vmax=2, cmap=cm)
    # #d_sc = ax2.scatter(expectation, scatter_d, s = 10, c = ratio, vmin=0, vmax=2, cmap=cm)
    # a_sc = ax1.scatter(expectation, scatter_a, s = 10, c = ratio, cmap=cm)
    # d_sc = ax2.scatter(expectation, scatter_d, s = 10, c = ratio, cmap=cm)
    # ax1.set_ylim([0,1])
    # ax2.set_ylim([0,0.1])
    # ax1.set_title('angle')
    # ax2.set_title('distance')
    # fig.colorbar(a_sc, ax=ax1)
    # fig.colorbar(d_sc, ax=ax2)
    # ax1.set_ylabel('error')
    # ax2.set_ylabel('error')
    # ax1.set_xlabel('expectation')
    # ax2.set_xlabel('expectation')
    # # ax1.set_yticklabels([])
    # #         ax2.set_yticklabels([])
    # # ax1.tick_params(length=0)
    # fig.savefig('./figure_h_map/r_i_{}.png'.format(r_i))
    # plt.close(fig)