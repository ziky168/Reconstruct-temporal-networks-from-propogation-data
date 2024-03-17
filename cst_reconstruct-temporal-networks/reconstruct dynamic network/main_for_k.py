# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 21:16:59 2024

@author: Rappy

<k> with SRAC...
"""
import pandas as pd
# import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
# from numba import jit
import time
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




    
# 参数设置
n_nodes = 100
p_all = np.arange(0.01,0.21,0.01)
time_steps = 5
num_simulations = 5000
n0 = 30

avg_results = pd.DataFrame(columns=['p', 'avg_SRAC', 'avg_SREL', 'avg_SRNC'])
for p in p_all:
    # 1 init ER for each time step.
    print("Now,p=",p)
    network = generate_networks.DynamicNetwork(n_nodes, p,'ER','Random')
    network.evolve(time_steps)

    model = SISModel(network,time_steps,num_simulations,n0=n0)
    # model = CPModel(network,time_steps,num_simulations,n0=n0)
    model.simulate()
    
    avg_SRAC = 0
    avg_SREL = 0
    avg_SRNC = 0
    for choose_time in range(time_steps):
        print("reconstruct T=",choose_time)
        G_reconstruct = reconstruction(model,choose_time)
        G = network.graph_history[choose_time]
        G_matrix = nx.to_numpy_matrix(G)
        G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
        _SRAC = SRAC(G_matrix,G_reconstruct_matrix)
        _SREL = SREL(G_matrix,G_reconstruct_matrix)
        _SRNC = SRNC(G_matrix,G_reconstruct_matrix)
        avg_SRAC += _SRAC
        avg_SREL += _SREL
        avg_SRNC += _SRNC
    avg_SRAC /= time_steps
    avg_SREL /= time_steps
    avg_SRNC /= time_steps
    # 将平均值存入avg_results数据框
    avg_results = avg_results.append({'p': p, 'avg_SRAC': avg_SRAC, 'avg_SREL': avg_SREL, 'avg_SRNC': avg_SRNC}, ignore_index=True)
    print("avg_SREL",avg_SREL)

avg_results.to_excel('average_results_SIS_ER.xlsx', index=False)


