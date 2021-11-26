# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:06:13 2020

@author: Administrator
"""

import math
import numpy as np
import scipy.stats as st

## --**--**--**--**--**--**--**--**-- 步骤1：定义交通异常态势检测函数 --**--**--**--**--**--**--**--**--**

# MAD函数
def MAD(seq):
    M = np.median(seq)
    mads = np.abs(seq-M)
    return np.median(mads)

# GESD函数    
def GESD(seq, alpha = 0.05, max_anoms = 0.2):
    r = math.ceil(len(seq)*max_anoms)
    lamda = [0]*r
    outlier_ind = [0]*r
    outlier_val = [0]*r
    seq_new = seq.copy()
    median_new = [0]*r
    mad_new = [0]*r
    ad_res = ["NO"]* len(seq) # 初始化结果，都非异常
    # 逐次查询最小值
    for i in range(0,r):       
        n = len(seq_new)
        median_new[i] = np.median(seq_new)
        mad_new[i] = MAD(seq_new)
        z = list((seq_new-np.median(seq_new))/(MAD(seq_new)+0.000001))
        max_ind = z.index(min(z))
        outlier_val[i] = seq_new[max_ind] # 新序列的最小值
        outlier_ind[i] = seq.index(seq_new[max_ind]) # 最小值在原序列中的索引
        del seq_new[max_ind]
        p = 1 - alpha/(n-i)
        t_pv = st.t.ppf(p, df = n-i-2)
        lamda[i] = (n-i-2)*t_pv/math.sqrt((n - i - 2 + t_pv**2) *(n - i))
        limit_lower = min(median_new[i]-lamda[i]*mad_new[i], 20) # 前提条件是处于拥堵状态
        # 检测到异常，更新结果序列a
        if(outlier_val[i] < limit_lower):
            ad_res[outlier_ind[i]] = "YES"
            for ii in range(0,i):
                ad_res[outlier_ind[ii]]="YES"
    return ad_res    

## --**--**--**--**--**--**--**--**-- 步骤2：检测检测核心流程 --**--**--**--**--**--**--**--**--**
'''
逐天进行数据检测；
以单个路段为基础单元，以5min为时间单位，计算平均速度，并进行索引（1~288）
根据历史同期的8个速度值，判断当前速度是否异常
如果为异常，则把 [时间索引、路段ID、速度值]写入本地anomalies_xx.txt文件
'''
import pyodbc
import time 
import datetime
import pandas as pd 

#----- 时间索引，按分钟
def time_ind(date):
    date1=time.strptime(str(date),"%Y%m%d%H%M%S")
    return date1[3]*60+date1[4]

#----- 当前日期前P个历史同期日期索引
def pre_day_week_ind(Y,M,D,P):
    now_time = datetime.date(Y,M,D)
    pre_time = now_time + datetime.timedelta(days=0-P*7)
    pre_time_format = pre_time.strftime('%Y%m%d')
    return pre_time_format

#----- 连接SQL
conn = pyodbc.connect(r'DRIVER={SQL Server}; SERVER=localhost; UID=IRC; PWD=IRC203; DATABASE=DongAo')
cursor = conn.cursor()

TD = 1 # 表示12月x日

#获取当天及历史同期时间索引
pre_days = [pre_day_week_ind(2019,12,TD,P) for P in range(0,9)] #当天和前8个相同day-of-week的日期标号

# 查询当天数据中的路段ID 
sql_1 = 'select distinct nds_id from autolr_{ymd}'.format(ymd = pre_days[0])
cursor.execute(sql_1)
ids = cursor.fetchall()
# 获取当天的路段列表
r_id =[]
for id in ids:
    r_id.append(id[0])

#----- 新建TXT，存放异常，每天的结果对应一个TXT
with open(r'C:\Users\Administrator\Desktop\anomaly_detection\risk_5_res\anomalies_' + str('%02d' % TD) + '.txt', 'a') as fw:  
    
    # ------ 遍历所有路段
    for i in range(0, len(r_id)): 
        sp_dict = {} # 定义字典：按天装载历史同期数据       
        # -----遍历收集历史同期的数据
        for d in pre_days:     
            #----- 筛选数据
            sql_2 = '''  select tm,speed from autolr_{m_d}
                      where nds_id = '{rid}'
                      order by tm '''.format(m_d = d, rid = r_id[i])
            try:          
                cursor.execute(sql_2)
                data = cursor.fetchall()
            except:
                continue
            
            #----- 1天数据归组
            sp_list = [None]*1440 # 创建空列表,原始数据1分钟1次记录，共1440行
            if not data: # data为空列表
                print("数据为空")
            else: # 非空
                for ele in data:
                    tm_ind = time_ind(ele[0])
                    sp_list[tm_ind] = int(ele[1])
            
            #----- 1天数据划分统计
            tt_seg = 5 # 时间窗
            
            unit = [ind for ind in range(0, 1440, tt_seg)] # 定义时间间隔，此为 5
            sp_unit = [] # 创建数据单元
            new_sp_list = [] 
            
            for num in unit:
                sp_unit = sp_list[num:(num + tt_seg)] #
                sp_unit = [e for e in sp_unit if e is not None] # 删除None
                if not sp_unit: # sp_unit为空
                    new_sp_list.append(None)
                else:           # sp_unit非空，将平均值加入新列表
                    sp_avg = int(np.mean(sp_unit))
                    new_sp_list.append(sp_avg)
            
            #----- 一天数据汇总
            sp_dict[d] = new_sp_list        
            #print(d + "完成！")
        print(str(i)+"："+ r_id[i] +"数据收集完成！")
        sp_df = pd.DataFrame(sp_dict)
        
        ## ----- 调用GESD(),异常检测
        for row in range(0,len(sp_df)):
            sp = list(sp_df.iloc[row])
            if ((sp[0] == None)):
                print("无数据:可能意味着道路封闭！")
                fw.write(str(row+1) + ',' + r_id[i] + ',' + 'NaN' + '\n')
            else:
                sp = [s for s in sp if s is not None]
                if((np.mean(sp[1:])-sp[0])>10): # 历史同期离散度
                    res = GESD(sp, alpha = 0.05, max_anoms = 0.2)
                    if(res[0]=="YES"): # 当天当前时段异常，保存
                        fw.write(str(row+1) + ',' + r_id[i] + ',' + str(sp[0]) + '\n')
                        print(sp, "-*-*- Anomaly -*-*-*")
        print(r_id[i]+"--AD检测完成！")

#----- 关闭SQL连接
cursor.close()
conn.close() 






