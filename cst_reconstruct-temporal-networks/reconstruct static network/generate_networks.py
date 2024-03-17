# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 17:01:11 2023

@author: Rappy
"""

import networkx as nx

#generate networks ER\BA\WS--------------------------
def generate_ER(nodenumber,p):
    #给节点数和连接概率p,返回ER网络
    G=nx.erdos_renyi_graph(nodenumber, p, seed=23, directed=False)
    return G

def generate_BA(nodenumber,edge_add):
    #给节点数和边增加数,返回BA网络
    G=nx.barabasi_albert_graph(nodenumber, edge_add,seed=23,directed=False)
    return G

def generate_WS(nodenumber,k,p):
    #给节点数,链接邻居数,断边重连数,返回WS网络
    G=nx.watts_strogatz_graph(nodenumber,k,p,seed=23)
    return G   
