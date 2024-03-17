# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:49:17 2024

@author: Rappy
generate-ER--
"""

import networkx as nx
from copy import deepcopy
import pandas as pd
import numpy as np


class DynamicNetwork:
    def __init__(self, n_nodes, p, networktype,evolveType):
        self.n_nodes = n_nodes
        self.p = p
        self.networktype = networktype
        self.evolveType = evolveType
        if self.networktype == "ER":
            self.graph = generate_ER(self.n_nodes, self.p)
        elif self.networktype == "BA":
            self.graph = generate_BA(self.n_nodes, int(self.p*self.n_nodes))
        elif self.networktype == "WS":
            self.graph = generate_WS(self.n_nodes, int(5), self.p)
        elif self.networktype == "ZK":
            self.graph = nx.karate_club_graph()
            self.n_nodes = nx.number_of_nodes(self.graph)
        else:
            raise ValueError("Invalid network type")
        self.graph_history = []  # 添加一个列表来存储网络结构的历史 
          
    def add_nodes(self):  
        self.graph.add_nodes_from(range(self.n_nodes))  
          
    def evolve_once(self):  
        # 网络演化
        if self.evolveType == "Random":
            if self.networktype == "ER":
                self.graph_history.append(deepcopy(self.graph))
                self.graph = nx.gnp_random_graph(self.n_nodes, self.p, directed=False)
            elif self.networktype == "BA":
                self.graph_history.append(deepcopy(self.graph))
                self.graph = nx.barabasi_albert_graph(self.n_nodes, int(self.p*self.n_nodes), seed=23)
            elif self.networktype == "WS":
                self.graph_history.append(deepcopy(self.graph))
                self.graph = nx.watts_strogatz_graph(self.n_nodes, int(5), self.p, seed=23)
            elif self.networktype == "ZK":
                self.graph_history.append(deepcopy(self.graph))
                self.graph = nx.karate_club_graph()
            elif self.networktype == "Miserables":
                self.graph_history.append(deepcopy(self.graph))
                self.graph = nx.les_miserables_graph()
            else:
                raise ValueError("Invalid network type")
        # elif self.evolveType == "Adaptive":
        #     #
        # elif self.evolveType == "BA":
        #     #
        else:
            raise ValueError("Invalid network type")
          
    def get_neighbors(self, node):  
        return list(self.graph.neighbors(node))
    
    
    def evolve(self, T):
        # 以self.graph为初始网络，以完全随机的方式进行网络演化（evolve_once）；在self.graph_history记录动态网络序列
        for t in range(T):
            self.evolve_once()
      
def generate_ER(nodenumber, p):
    #给节点数和连接概率p,返回ER网络
    G = nx.erdos_renyi_graph(nodenumber, p, seed=23, directed=False)
    return G

def generate_BA(nodenumber, edge_add):
    #给节点数和边增加数,返回BA网络
    G = nx.barabasi_albert_graph(nodenumber, edge_add, seed=23)
    return G

def generate_WS(nodenumber, k, p):
    #给节点数,链接邻居数,断边重连数,返回WS网络
    G = nx.watts_strogatz_graph(nodenumber, k, p, seed=23)
    return G

# # simulated Dynamic network finish
class Real_DynamicNetwork:
    def __init__(self, path, maxtime = 30):
        if path == 'email-Eu-core-temporal-Dept4.txt' or path == 'email-Eu-core-temporal-Dept3.txt':
            df = pd.read_csv(path, delimiter='\s+').to_numpy()
            df[:,2]=df[:,2]/86400 # dim=2 time /per day
            self.graph_history = []  # 添加一个列表来存储网络结构的历史 
            self.maxtime = min(maxtime,np.max(df[:,2]))
            self.n_nodes = max(np.max(df[:,0]),np.max(df[:,1]))+1
            for t in range(self.maxtime + 1):
                G = nx.erdos_renyi_graph(self.n_nodes, 0, seed=23, directed=False)
                edges = df[df[:, 2] <= t]
                for edge in edges:
                    source, target, _ = edge
                    G.add_edge(source, target)
                self.graph_history.append(G)
        elif path == 'ca-netscience.txt':
            df = pd.read_csv(path, delimiter='\s+').to_numpy()
            self.graph_history = []
            self.maxtime = maxtime
            self.n_nodes = max(np.max(df[:,0]),np.max(df[:,1]))
            G = nx.erdos_renyi_graph(self.n_nodes, 0, seed=23, directed=False)
            df = df - 1 # (1~100) to (0~99)
            df = np.where(df < self.n_nodes, df,None) # sample
            edges = [(source, target) for (source, target) in df if source is not None and target is not None]
            for edge in edges:
                source, target = edge
                G.add_edge(source, target)
            for t in range(self.maxtime + 1):
                self.graph_history.append(G)
        else:
            raise ValueError("Invalid network file")
        