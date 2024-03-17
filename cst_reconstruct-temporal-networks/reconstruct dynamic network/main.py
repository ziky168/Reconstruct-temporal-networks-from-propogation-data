# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:46:48 2024

@author: Rappy
"""
import pandas as pd
# import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
# import csv
import time
# import os
# import math
# from scipy.optimize import minimize
# from scipy.optimize import Bounds
import logging
# from datetime import datetime
import networkx as nx
from Spreading import SISModel,CPModel
import generate_networks
from Reconstruct import reconstruction
# from multiprocessing import Pool
from evaluate_accuracy import SRAC,SREL,SRNC,CR
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


if __name__ == '__main__':

    
#     # 参数设置  
    n_nodes = 500 #结点数目
    p = 0.03 # ER:p BAk=n_nodes*p
    time_steps = 5 
    num_simulations = 5000
    n0 = 30
    
    # 1 init ER for each time step.
    print("generate network start----------")
    print("time steps:",time_steps)
    spread_time0 = time.time()
    network = generate_networks.DynamicNetwork(n_nodes, p,'ER','Random') # Generative dynamic networks...#ER\BA\WS\ZK...
    network.evolve(time_steps)
    
    # network = generate_networks.Real_DynamicNetwork('ca-netscience.txt') # If you want reconstruct real dynamic networks...
    spread_time1 = time.time()
    print("generating dynamic networks cost(s):",spread_time1-spread_time0)
    
    # G = nx.read_gexf('path_to_file.gexf')     
    # 2 spread
    # print("simulate spreading process start-------")
    spread_time0 = time.time()
    
    # model = SISModel(network,time_steps,num_simulations,n0=n0)
    model = CPModel(network,time_steps,num_simulations,n0=n0)
    model.simulate()
    
    spread_time1 = time.time()
    print("spreading simulated time(s):",spread_time1-spread_time0)
    # print("simulate spreading process finished-------")
    
    # 3 reconstruct
    # print("reconstruct strat-----------")
    choose_time = 3
    print("reconstruct T=",choose_time)
    reconstruct_time0 = time.time()
    
    G_reconstruct = reconstruction(model,choose_time)
    
    # print("reconstruct_nodes number:",G_reconstruct.number_of_nodes())
    # print("reconstruct_edges number:",G_reconstruct.number_of_edges())
    reconstruct_time1 = time.time()
    ## save
    G = network.graph_history[choose_time]
    nx.write_adjlist(G, "G.adjlist")
    nx.write_adjlist(G_reconstruct, "G_reconstruct.adjlist")
    
    # 4 visual
    # 对比...图,邻接矩阵
    fig = plt.figure(figsize=(8,8),dpi=400)
    plt.suptitle("your title")
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

    
    # 计算度分布  
    degree_G0 = nx.degree(G)  
    degree_G1 = nx.degree(G_reconstruct)  
      
    # 计算聚类系数分布  
    clustering_G0 = nx.clustering(G)  
    clustering_G1 = nx.clustering(G_reconstruct)  
      
    # 计算介数中心性  
    betweenness_G0 = nx.betweenness_centrality(G)  
    betweenness_G1 = nx.betweenness_centrality(G_reconstruct)  
      
    # 计算接近中心性  
    closeness_G0 = nx.closeness_centrality(G)  
    closeness_G1 = nx.closeness_centrality(G_reconstruct)  
    
      
    # 绘制度分布图  
    plt.figure(figsize=(12, 12))  
    
    # 度分布
    plt.subplot(4, 1, 1)
    degree_dist_G0 = [degree for node, degree in degree_G0]
    degree_dist_G1 = [degree for node, degree in degree_G1]
    plt.hist(degree_dist_G0, bins=np.arange(max(degree_dist_G0)+2)-0.5, alpha=0.5, color='b', label='G')
    plt.hist(degree_dist_G1, bins=np.arange(max(degree_dist_G1)+2)-0.5, alpha=0.5, color='r', label='G_reconstruct')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    # 聚类系数分布图
    plt.subplot(4, 1, 2)
    plt.hist(list(clustering_G0.values()), bins=20, alpha=0.5, color='b', label='G_origin')
    plt.hist(list(clustering_G1.values()), bins=20, alpha=0.5, color='r', label='G_reconstruct')
    plt.title('Clustering Coefficient Distribution')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    # 介数中心性分布图
    plt.subplot(4, 1, 3)
    plt.hist(list(betweenness_G0.values()), bins=20, alpha=0.5, color='b', label='G_origin')
    plt.hist(list(betweenness_G1.values()), bins=20, alpha=0.5, color='r', label='G_reconstruct')
    plt.title('Betweenness Centrality Distribution')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')
    plt.legend()
    # 接近中心性分布图  
    plt.subplot(4, 1, 4)
    plt.hist(list(closeness_G0.values()), bins=20, alpha=0.5, color='b', label='G_origin')
    plt.hist(list(closeness_G1.values()), bins=20, alpha=0.5, color='r', label='G_reconstruct')
    plt.title('Closeness Centrality Distribution')
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()  
    plt.show()

    # 5 Assess accuracy
    # # if you want to read saved graph
    # G = nx.read_adjlist("G.adjlist")
    # G_reconstruct = nx.read_adjlist("G_reconstruct.adjlist")
    
    print("reconstruct simulated time(hr):",(reconstruct_time1-reconstruct_time0)/3600)
    G_matrix = nx.to_numpy_matrix(G)
    G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
    print("SRAC:",SRAC(G_matrix,G_reconstruct_matrix))
    print("SREL:",SREL(G_matrix,G_reconstruct_matrix))
    print("SRNC:",SRNC(G_matrix,G_reconstruct_matrix))
    print("CR:",CR(G_reconstruct_matrix))
    
    # # # reconstructing with t*
    # # print("reconstruct strat-----------")
    # results = pd.DataFrame(columns=['choose_time', '_SRAC', '_SREL', '_SRNC'])
    # reconstruct_times0 = time.time()
    # for choose_time in range(time_steps):
    #     print("reconstruct T=",choose_time)
    #     G_reconstruct = reconstruction(model,choose_time)
    #     G = network.graph_history[choose_time]
    #     G_matrix = nx.to_numpy_matrix(G)
    #     G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
    #     _SRAC = SRAC(G_matrix,G_reconstruct_matrix)
    #     _SREL = SREL(G_matrix,G_reconstruct_matrix)
    #     _SRNC = SRNC(G_matrix,G_reconstruct_matrix)
    #     results = results.append({'choose_time': choose_time, '_SRAC': _SRAC, '_SREL': _SREL, '_SRNC': _SRNC}, ignore_index=True)
    #     print("SRAC:",_SRAC)
    #     print("SREL:",_SREL)
    #     print("SRNC:",_SRNC)
    # reconstruct_times1 = time.time()
    # print("reconstruct simulated time(hr):",(reconstruct_times1-reconstruct_times0)/3600)
    # results.to_excel('results_SIS_BA.xlsx', index=False)
    
    
    