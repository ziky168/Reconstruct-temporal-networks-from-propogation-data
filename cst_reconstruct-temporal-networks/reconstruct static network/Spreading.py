# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:02:36 2023

@author: Rappy
"""
import numpy as np
from numba import jit
import copy

dynamic_network = False # 是否是时序网络；若是,则True.

def spreading(G,T,N,n0):
    #给定网络G和传播时间T,模拟传播过程,n0为初始感染人数。
    series = np.zeros([N,T.size+1], dtype = bool) # 记录状态
    global lambdai
    lambdai = np.random.uniform(0.2,0.4,N) # 个人感染率 (0.2~0.4) in SIS dynamics;(0.7~0.9)in CP
    global deltai  
    deltai = np.random.uniform(0.4,0.6,N) # 个人恢复力,染病后恢复率 (0.4~0.6) in SIS dynamics;(0.2~0.4)in CP
    #初始化节点状态
    LList = range(0,N)
    series0 = list(np.random.choice(LList, n0,replace=False)) # 最初感染源头
    series[series0, 0] = 1 # 修改状态记录第一列数据
    #迭代传播动力学,记录新状态
    infecting = set(series0) # 初始
    for t in T:
        #感染过程
        infecting0 = copy.deepcopy(infecting)
        infecting = propagation01(G,infecting0)
        series[list(infecting), t] = 1
        # series[:,:,times] = seriesonetime
        #康复过程
        recovered = propagation10(infecting0)
        series[list(recovered), t] = 0
        infecting = infecting - recovered
        
    return series

def propagation01(G,lastinfected_nodes):
    #从0到1,节点从未感染到感染过程
    neighbours = set() # 密接节点
    newinfected = set() # 新时刻感染者
    for inf in lastinfected_nodes:
        neighbours.update(set(G[inf])) # 密接节点
    neighbours = neighbours.difference(lastinfected_nodes)
    for node in list(neighbours):
        if prob01(G,lastinfected_nodes,node) > np.random.random(): #若 感染
            newinfected.add(node)
    newinfected.update(lastinfected_nodes)
    return newinfected

def prob01(G,lastinfected_nodes,node):
    # 返回节点的感染概率Pn
    # neigh_infect = np.sum(np.isin(G[node], list(lastinfected_nodes)))  # 比下面这行要慢。
    neigh_infect = len(set(G[node]) & lastinfected_nodes) # 节点认识的人中有多少被感染的；
    Pn=1-pow(1-lambdai[node],neigh_infect) # 节点 node 的感染概率
    return Pn

def propagation10(lastinfected):
    #从1到0,康复过程;康复后可再次被感染;
    recovered = set()
    for node in list(lastinfected):
        if deltai[node] > np.random.random():
            recovered.add(node)
    return recovered