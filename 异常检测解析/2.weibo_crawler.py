# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:22:38 2020

@author: senda
"""

import time
import xlrd
import xlwt
from xlutils.copy import copy
from selenium import webdriver

def write_excel_xls(path, sheet_name, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j])  # 像表格中写入数据（对应的行和列）
    workbook.save(path)  # 保存工作簿
    print("新建xls格式表格+写入数据成功！")

def write_excel_xls_append(path, value):
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i+rows_old, j, value[i][j])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")

## 加载微博页面         
driver = webdriver.Chrome(r'D:\360安全浏览器下载\chromedriver') #存放驱动的位置
driver.set_window_size(1400, 800) #指定打开浏览器的大小
driver.get("https://s.weibo.com/") #微博高级搜索网址
time.sleep(2)

## 自定义参数
username = "15239035446" #你的微博登录名
password = "xxxxxxx" #你的密码
keywords = "北京 交通事故" #输入你想要的关键字，建议有超话的话加上##，如果结果较少，不加#

## 登录
driver.find_element_by_xpath("//*[@node-type='loginBtn']").click() # 点击登录获得登录界面

elem = driver.find_element_by_name("username");  # 用户名
elem.send_keys(username)
elem = driver.find_element_by_name("password");  # 密码
elem.send_keys(password)
driver.find_element_by_xpath("//*[@node-type='submitBtn']").click() # 点击登录

## 关键字搜索
elem = driver.find_element_by_xpath("//*[@node-type='text']")
elem.send_keys(keywords)
driver.find_element_by_xpath("//*[@node-type='submit']").click() # 点击搜索

######### ------------- 手动进行高级搜索（确定时空范围） ------------##########
######### ------------- 手动进行高级搜索（确定时空范围） ------------##########

## 内容存储
sheet_name_xls = '微博数据' #sheet表名
value_title = [["用户名称","微博内容", "发布时间"],]
#填写你想存放excel的路径，没有文件会自动创建
book_name_xls = r"C:\Users\g\Desktop\anomaly detection\weibo_data\train\sport\201912.xls" 
write_excel_xls(book_name_xls, sheet_name_xls, value_title)

#获取信息
while True:
    elems = driver.find_elements_by_xpath("//*[@action-type='feed_list_item']")
    for i in range(0,len(elems)):
        try:
            w_username = elems[i].find_elements_by_css_selector('a.name')[0].text
            #w_content = elems[i].find_element_by_xpath("//*[@node-type='feed_list_content']").text
            w_content = elems[i].find_elements_by_css_selector("p.txt")[0].text                            
            w_time = elems[i].find_elements_by_css_selector("p.from > a[target='_blank']")[0].text
            #w_share = elems[0].find_element_by_xpath("//*[@action-type='feed_list_forward']").text
            #w_comment = elems[0].find_element_by_xpath("//*[@action-type='feed_list_comment']").text
        except:
            continue
        value1 = [[w_username,w_content,w_time],]
        write_excel_xls_append(book_name_xls, value1)
    # 加载下一页
    try:
        driver.find_element_by_css_selector('a.next').click() #
        driver.implicitly_wait(20) #
    except:
        print("##--##--##--##--全部加载完成--##--##--##--##")
        break # 无下一页，退出循环
