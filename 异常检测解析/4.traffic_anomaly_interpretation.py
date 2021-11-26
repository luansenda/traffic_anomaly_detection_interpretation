# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:44:41 2020

@author: senda
"""

##----------------------------- 读取路网基础信息：{路段索引、路段断点坐标} ----------------------------##
import re 
road_segs = {}
rid_dict = {}
rid_ind = 0
with open(r'C:\Users\Administrator\Desktop\anomaly_detection\link_info_201912.txt','r', encoding = 'utf-8') as fopen:
    fopen.readline() # 读掉标题行
    for line in fopen:
        ll = line.split('\t')
        nodes = re.findall(r'\d+.\d+',ll[-1]) # 从坐标串中寻找浮点数（格式：xxx.xxx）
        road_segs[ll[0]]=[[float(nodes[0]),float(nodes[1])],[float(nodes[-2]),float(nodes[-1])]]
        rid_ind += 1
        rid_dict[str(rid_ind)]=ll[0]

rid_dict_r = {v:k for k,v in rid_dict.items()}

#import json
#jsdata = json.dumps(road_segs,ensure_ascii=False)
#with open(r'C:\Users\Administrator\Desktop\anomaly_detection\roadsegs_11.js','w+') as fw:
#    fw.write('var roadsegs =')
#    fw.write(jsdata)

## ---------------------------------- 遍历某一天的结果,构建异常树 --------------------------------------##    
day = 3 # 12月x日的结果
import pyodbc
## ---------------------- 新建表，用来存放树枝
tree_table_name = 'trees_day'+str(day)
conn = pyodbc.connect(r'DRIVER={SQL Server}; SERVER=localhost; UID=IRC; PWD=IRC203; DATABASE=NRC_Trees')
cursor = conn.cursor() 
sql1 = '''IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID('{0}')) 
            BEGIN
            CREATE TABLE {0}(
             start_t int,
             tree	NVARCHAR(1000) NOT NULL)
            END
            '''        
cursor.execute(sql1.format(tree_table_name))   
conn.commit() # 若数据库有修改变动，需要commit()
cursor.close()
conn.close() 

                                  
##---------------------------- 读取本地的异常检测结果 
import os
filepath = r'C:\Users\Administrator\Desktop\anomaly_detection\risk_5_res'  # 文件夹路径
filename = 'anomalies_' + '%02d' % day + '.txt'  # 文件名
datapath = os.path.join(filepath, filename)    # 数据文件路径

time_win = 5 # 设置时间间隔
snapshots = 1440//time_win #整数型

NRCres = {} # 空字典，用于存放不同时间片的异常路段
for i in range(1,snapshots+1): 
    NRCres[i] = []    
    
with open(datapath, 'r', encoding = 'utf-8') as ff:
    for line in ff:
        line = line.strip('\n')
        ll = line.split(',')
        if(ll[2] != 'NaN'):
            NRCres[int(ll[0])].append([ll[1]])

##------------------------------ 构建异常树状结构 
NRCTrees = {}
pre_Twindow = []

for i in range(1,snapshots+1): # traverse each time-window in order
    # 读取当前time-window的NRCs
    NRCs = NRCres[i]     
    # 若NRCs为空，，，，转入next time-window
    if(len(NRCs) == 0): 
        NRCTrees[i] = []
        pre_Twindow = []
        continue    
    # 首次遇到NRCs不为空，添加NRC节点，转入next time-window
    if(len(pre_Twindow) == 0): 
        NRCTrees[i] = NRCs
        pre_Twindow = NRCs
        continue
    
    used_trees = [] # 存储与next time-window中NRC关联的节点/树枝
    new_Twindow = [] # 存储新形成的树枝    
    # 构建树枝,遵循（1）空间可达性，（2）时间有效性
    for NRC in NRCs:
        new_trees = 0 # 可构建树枝的数量初始值为0
        for tree in pre_Twindow:
            up_out = road_segs[NRC[0]][1]
            down_in = road_segs[tree[-1]][0]
            try:
                if(up_out == down_in): # 如果上游路段的出口与下游路段的入口一致，即，认为空间可达
                    used_trees.append(tree)
                    new_tree = tree + NRC # 增加新节点，更新new_tree
                    new_Twindow.append(new_tree) # 保存new_tree
                    new_trees += 1 # 找到有效子节点，树枝量+1
            except:
                continue
        if(new_trees == 0): # 如果NRC没有找到一个可关联的父节点,直接存储当前NRC,可作为下一snapshot中的潜在父节点
            new_Twindow.append(NRC)
        
    NRCTrees[i] = new_Twindow # 存储新树到当前time-window
    #NRCTrees[i-1] = [e for e in pre_Twindow if e not in used_trees]  # pre_Twindow中被用到的节点/树枝将不被保留
    NRCTrees[i-1] = [e for e in pre_Twindow if (e not in used_trees and len(e)>4)]  # pre_Twindow中被用到的节点/树枝,以及单个节点将不被保留
    pre_Twindow = new_Twindow # 迭代 
NRCTrees[i] = [e for e in pre_Twindow if (e not in used_trees and len(e)>4)]  # 最后处理时间片i

## -------------------- 提取一天的树枝，导入SQL --------------------##
conn = pyodbc.connect(r'DRIVER={SQL Server}; SERVER=localhost; UID=IRC; PWD=IRC203; DATABASE=NRC_Trees')
cursor = conn.cursor() 

sql2 = '''insert into {0} values({1},'{2}')'''   

for tt in NRCTrees:
    for tr in NRCTrees[tt]:
        new_list = [rid_dict_r[rid] for rid in tr]
        tree = '-'.join(new_list) #拼接rid
        cursor.execute(sql2.format(tree_table_name,tt,tree))    
        conn.commit() # 若数据库有修改变动，需要commit(),
cursor.close()
conn.close() 
            
##--------------------------------------- 从SQL中选择最大化的树枝 -----------------------------------------##
'''
从数据库中处理：
树枝数量大于5，
合并时间相近、相似度大于0.6的树枝
'''

import pyodbc
conn = pyodbc.connect(r'DRIVER={SQL Server}; SERVER=localhost; UID=IRC; PWD=IRC203; DATABASE=NRC_Trees')
cursor = conn.cursor() 
sql3 = '''SELECT * from {0} order by start_t'''        
cursor.execute(sql3.format(tree_table_name)) 
rid_strs = cursor.fetchall()  
cursor.close()
conn.close() 

rid_str = []
for tu in rid_strs:
    rid_str.append({'time':tu[0],
                    'net': tu[1]})

## ------------------------------------找最大的子网络    
rid_str_max = []
for r in range(0,len(rid_str)):
    flag1 = 0
    flag2 = 0
    ## 先与r_str_max进行判断
    for r_max in rid_str_max: 
        if rid_str[r]['net'] in r_max: # 如果当前树枝属于r_str_max中的树枝的子集，则跳过这个循环
            flag1 = 1
            break
    if flag1 == 1: 
        continue
    
    ## 后与r后面剩余的进行判断
    for rr in range(r+1,len(rid_str)):
        if rid_str[r]['net'] in rid_str[rr]['net']: # 如果当前树枝属于r_str_max中的树枝的子集，则跳出这个循环
            flag2 = 1
            break
    if flag2 == 0: # 若flag2无变化，则当前树枝最长，添加到rid_str_max
        rid_str_max.append(rid_str[r])

## -------------------------------- 根据相似度，合并相似的树枝
## 方法1
import difflib
def sim_ratio(s1,s2):
    return difflib.SequenceMatcher(None,s1,s2).ratio()
## 方法2
#import Levenshtein
#def sim_ratio2(s1,s2):
#    return Levenshtein.ratio(s1,s2)
    
## --------------基于相似度的无监督分类
net_cate = {}
cate = 0 #初始类别数

for sub_net in rid_str_max:
    flag = 0
    if not net_cate: # 如果为空
        cate += 1
        net_cate[cate] = [sub_net]
        continue
    for key in net_cate:
        if sim_ratio(sub_net['net'], net_cate[key][0]['net']) > 0.6: # 与每一类的第一项进行相似性比较
            net_cate[key].append(sub_net)
            flag = 1
            break
    if flag == 0: #如果没有从已有类中找到相似项，则新建类
        cate += 1
        net_cate[cate] = [sub_net]
    
## --------------合并,并按索引转换为rid, 保存检测时间+合并的异常树枝
rid_net_integrate = []
for key in net_cate:
    
    for ele in net_cate[key]:
        max_len = 0
        max_t = 0
        ll = ele['net'].split('-')
        if (len(ll)>5 and len(ll)>max_len):
            max_len = len(ll)
            rid_net = ele['net']
            max_t = ele['time']
    if(max_t !=0):
        rid_net=rid_net.split('-')
        rids = [rid_dict[ind] for ind in rid_net]
        Day = "%02d" % day 
        Hour = "%02d" % (max_t*5//60)
        Minite = "%02d" % (max_t*5%60)
        rid_net_integrate.append({'detect_t':'2019-12-{Day} {hour}:{Min}:00'.format(Day=Day, hour=Hour, Min= Minite),
                                  'rids':rids}) 
        #print('2019-12-{Day} {hour}:{Min}:00'.format(Day=Day, hour=Hour, Min= Minite),rids)
        


## -------------------------------------- 基于交通事件日志的验证 --------------------------##
        
'''
时空匹配
时间偏差60 min
空间偏差1000 m
'''
## ---------------定义距离计算函数
import math
def dist_calc(olon, olat, dlon, dlat):
    # gps points
    CurLong = olon
    CurLat = olat
    DesLong = dlon
    DesLat = dlat
    a = 6378137
    b = 6356752
    # 坐标转换弧度
    Curlamda = CurLong*math.pi/180
    Curfai = CurLat*math.pi/180
    Deslamda = DesLong*math.pi/180
    Desfai = DesLat*math.pi/180
    # O点
    la = a*math.cos(Curfai)
    lb = b*math.sin(Curfai)
    lc = math.sqrt(la*la + lb*lb) # O点球心距
    CurN =(a*b)/lc
    CurX = CurN * math.cos(Curfai)*math.cos(Curlamda)
    CurY = CurN * math.cos(Curfai)*math.sin(Curlamda)
    CurZ = CurN * b*b/a/a*math.sin(Curfai)
    # D点
    la = a*math.cos(Desfai)
    lb = b*math.sin(Desfai)
    lc = math.sqrt(la*la + lb*lb) # D点球心距
    DesN =(a*b)/lc
    DesX = DesN *math.cos(Desfai)*math.cos(Deslamda)
    DesY = DesN *math.cos(Desfai)*math.sin(Deslamda)
    DesZ = DesN*b*b/a/a*math.sin(Desfai)
    # OD
    la = DesX - CurX
    lb = DesY - CurY
    lc = DesZ - CurZ
    # distance
    Dis = math.sqrt(la*la + lb*lb + lc*lc)
    return round(Dis, 2)

## ---------------定义时间差计算函数
from datetime import datetime

def timediff(time_1,time_2):
    time_1_struct = datetime.strptime(time_1, "%Y-%m-%d %H:%M:%S")
    time_2_struct = datetime.strptime(time_2, "%Y-%m-%d %H:%M:%S")
    seconds = (time_2_struct - time_1_struct).seconds
    min_diff = seconds/60
    return round(min(min_diff,1440-min_diff),2)


## ---------------提取UGC
conn = pyodbc.connect(r'DRIVER={SQL Server}; SERVER=localhost; UID=IRC; PWD=IRC203; DATABASE=NRC_Trees')
cursor = conn.cursor() 
sql4 = '''
        SELECT [start_date]
              ,[end_date]
              ,[lng]
              ,[lat]
              ,[event_desc]
              ,[nds_ids]
          FROM [DongAo_ugc].[dbo].[ugc_201912_merge]
          where start_date like '2019-12-{Day}%' --and evt_type_sub_id != 4
          order by start_date'''        
cursor.execute(sql4.format(Day=Day)) 
ugcs = cursor.fetchall()  
cursor.close()
conn.close()     

## ----------------时空匹配
incid_ind = 0
for net in rid_net_integrate:
    time_flag = False
    dist_flag = False

    for ugc in ugcs:
        ## 时间
        t1 = timediff(net['detect_t'],ugc[0])
        t2 = timediff(net['detect_t'],ugc[1])
        if(t1<45 or t2<45): # 时间偏差45min
            time_flag = True
        else:
            continue
        
        ## 空间距离   
        ugc_lng = ugc[2]
        ugc_lat = ugc[3]           
        dist_list = []
        for rid in net['rids']:
            rid_lng = road_segs[rid][0][0]
            rid_lat = road_segs[rid][0][1]         
            dist = dist_calc(float(ugc_lng),float(ugc_lat),rid_lng,rid_lat)
            dist_list.append(dist)
            
        if(min(dist_list) <1000): # 距离偏差1000米
            dist_flag = True
        
        ## 时空偏差阈值均满足
        if (time_flag and dist_flag):
            incid_ind += 1
            print('%03d' % incid_ind,'：--- success ---', "交通异常与交通事件的时空匹配{",'时间偏差：',min(t1,t2),' && ','距离偏差：',min(dist_list), '}')
            break
        
    ## 时空偏差阈值有一个不满足，则无法验证
    if (time_flag == False or dist_flag==False):
        incid_ind += 1
        print('%03d' % incid_ind,'-*-*-*-*-*-*-*-*-*-*-*-*-*-*- the verification is failure -*-*-*-*-*-*-*-*-*-*-*-*-*-*-')
    
    



    
    
    
    
    
    
