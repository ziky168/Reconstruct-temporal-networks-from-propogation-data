# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 16:32:21 2023

@author: Rappy
"""
import pandas as pd
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import csv
import time
import os
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
import logging
from datetime import datetime
import networkx as nx
from Spreading import spreading
import generate_networks
from Reconstruct import reconstruction
from multiprocessing import Pool
from evaluate_accuracy import SRAC,SREL,SRNC,CR
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def run_spreading_simulation(args):
    G, T, N, n0 = args
    return spreading(G, T, N, n0).T

if __name__ == '__main__':

    
    #initial
    T_final = 5000 # 传播总时长
    T_step = 1 # 传播记录时间步
    T = np.arange(1, T_final,T_step, int)
    N = int(500) # 节点数目
    n0 = 2 #初始感染节点个数
    M = 10 #拼接10次传播

    
    # 1 init
    # WS / NW / ER / BA 
    print("generate network start----------")
    p = 0.03
    G = generate_networks.generate_ER(N,p)
    # G = generate_networks.generate_WS(N,10,p)
    print("nodes number:",G.number_of_nodes())
    print("edges number:",G.number_of_edges())
    
    # 2 spread
    spread_time0 = time.time()
    print("simulate spreading process start-------")
    
    pool = Pool()
    args_list = [(G,T,N,n0) for _ in range(M)]
    results = pool.map(run_spreading_simulation,args_list)
    pool.close()
    pool.join()
    infect_T = np.concatenate(results,axis = 0)
    
    # infect_T = spreading(G,T,N,n0).T  #shape=(T,N)
    plt.plot(np.sum(infect_T,axis=1))
    # for times in range(M):
    #     infect_Ti = spreading(G,T,N,n0).T
    #     infect_T = np.concatenate((infect_T,infect_Ti),axis=0)
    #infect_T matrix shape:T*N
    print('time_step:',infect_T.shape[0])
    print('nodes_number:',infect_T.shape[1])
    spread_time1 = time.time()
    print("spreading simulated time:",spread_time1-spread_time0)
    print("simulate spreading process finished-------")
    
    # 3 reconstruct
    print("reconstruct strat-----------")
    reconstruct_time0 = time.time()
    G_reconstruct = reconstruction(infect_T)
    print("reconstruct_nodes number:",G_reconstruct.number_of_nodes())
    print("reconstruct_edges number:",G_reconstruct.number_of_edges())
    reconstruct_time1 = time.time()
    print("reconstruct simulated time(hr):",(reconstruct_time1-reconstruct_time0)/60)
    ## save 
    nx.write_adjlist(G, "G.adjlist")
    nx.write_adjlist(G_reconstruct, "G_reconstruct.adjlist")
    
    # 4 visual
    fig = plt.figure(figsize=(8,8),dpi=400)
    ax1 = fig.add_subplot(2,2,1)
    nx.draw_networkx(G,pos = nx.spring_layout(G,seed=11),with_labels = False,arrows=None,node_size=20,node_color="tab:red",edge_color="tab:gray")
    ax1.set_title('G_origin')
    
    ax2 = fig.add_subplot(2,2,2)
    nx.draw_networkx(G_reconstruct, pos=nx.spring_layout(G_reconstruct,seed=11),with_labels = False,arrows=None,node_size=20,node_color="tab:blue",edge_color="tab:gray")
    ax2.set_title('G_reconstructed')
    
    ax3 = fig.add_subplot(2,2,3)
    ax3.matshow(nx.to_numpy_matrix(G),cmap=plt.cm.gray)
    
    ax4 = fig.add_subplot(2,2,4)
    ax4.matshow(nx.to_numpy_matrix(G_reconstruct),cmap=plt.cm.gray)
    
    plt.show()
    
    # 5 Assess accuracy
    # G = nx.read_adjlist("G.adjlist")
    # G_reconstruct = nx.read_adjlist("G_reconstruct.adjlist")
    G_matrix = nx.to_numpy_matrix(G)
    G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
    print("SRAC:",SRAC(G_matrix,G_reconstruct_matrix))
    print("SREL:",SREL(G_matrix,G_reconstruct_matrix))
    print("SRNC:",SRNC(G_matrix,G_reconstruct_matrix))
    print("CR:",CR(G_reconstruct_matrix))
    
    
    