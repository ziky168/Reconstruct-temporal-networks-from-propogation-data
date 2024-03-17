# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:03:40 2023

@author: Rappy
"""
import numpy as np
import cvxpy as cvx
from scipy.spatial.distance import hamming
from scipy.spatial.distance import cdist
import networkx as nx
import generate_networks
import matplotlib.pyplot as plt
from numba import jit

def reconstruction(infect_T):
    N = infect_T.shape[1]
    G_reconstruct = generate_networks.generate_ER(N,0)
    # plt.figure(dpi=400)
    # plt.ion()
    for i in range(N):
        [Y,Phi] = findYetPhi(i,infect_T)
        [connected,x] = get_Edge(Y,Phi,i)
        edge_list = list(map(lambda t:(i,t),list(connected)))
        G_reconstruct.add_edges_from(edge_list) # G.add_edges_from([(1, 2), (1, 3)])
        # reconstruct_scatter(x*(-1),G,i)
        # plt.pause(0.1)
    # plt.legend()
    # plt.ioff()
    # plt.show()
    return G_reconstruct

# @jit(nopython=True)
def get_Edge(Y,Phi,i):
    [m,n] = Phi.shape #n=N-1
    min0 = 1e-1 #分离阈值
    if m<n:
        # #凸规划最小化x的范数1 CST解欠定方程组
        x = cvx.Variable(n)
        objective = cvx.Minimize(cvx.norm(x, 1))
        constraints = [Phi @ x == Y]
        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        if x.value is None: 
            connected = []
            # print(x.value)
        else:
            connected = np.where(x.value*(-1) >= min0) # a_ij != 0
            # print(x.value)
            connected = np.where(connected[0]>=i,connected[0]+1,connected[0])
    # elif m==n:
    #     # #正定方程组
    #     x = np.linalg.solve(Phi, Y)
    #     connected = np.where(x[0]*(-1) >= min0) 
    #     connected = np.where(connected[0]>=i,connected[0]+1,connected[0])
    # else:
    #     # #超定方程组
    #     x = np.linalg.lstsq(Phi, Y)
    #     connected = np.where(x[0]*(-1) >= min0) 
    #     connected = np.where(connected[0]>=i,connected[0]+1,connected[0])
    return [connected,x.value]


def findYetPhi(i,seriesTN):
    # seriesTN : columns are time dimensions
    # function : find Y and Phi based on i
    N = seriesTN.shape[1]
    # a) find 00 or 01 : base_rows, from this row to the next, the state of i is from 0 to 0/1
    Sit0 , Sothers_t0 = select_data(i,seriesTN)
    
    # b) Delta test and Theta test
    base_strings = get_base_strings(Sit0,Sothers_t0)

    # c) got Y and Phi
    # restrain nt
    nt = 0.8
    rownt = int(nt*(N-1))
    if len(base_strings) > rownt:
        base_strings = base_strings[:rownt]
    epsilon = 0
    
    # Convert S_it and Sothers_t0 to int if they are not already
    Sit0 = Sit0.astype(int)
    Sothers_t0 = Sothers_t0.astype(int)
    
    # Compute Y using list comprehension
    Y = [np.log(1 + epsilon - np.average(Sit0[base], axis=0)) for base in base_strings]
    Y = np.asarray(Y)
    # Compute Phi using list comprehension
    Phi = [np.average(Sothers_t0[base], axis=0) for base in base_strings]
    Phi = np.asarray(Phi)

    return [Y,Phi]

def select_data(i,seriesTN):
    
    S_it = seriesTN[:,i] # The state of the node i at each moment; shepe=[T.size,1]
    Sothers_t = np.delete(seriesTN,i,1) # the state of other nodes rather than i at each moment; shape = [T.size,N-1]
    rows0 = set(np.argwhere(S_it == 0).flatten())
    rows0.discard(len(S_it)-1) #ignore the last 0.
    effectrows = rows0
    effectrowsnext = set(map(lambda x:x+1,effectrows)) # following strings
    Sit0 = S_it[list(effectrowsnext)]
    Sothers_t0 = Sothers_t[list(effectrows)]
    
    return Sit0, Sothers_t0

def get_base_strings(Sit0,Sothers_t0):
    normalized_hamming_matrix = cdist(Sothers_t0, Sothers_t0, metric='hamming')
    # divide different base strings > Theta
    Theta = 0.25 #0.25 in SIS; 0.35 in CPs
    Delta = 0.45 #0.45 in SIS; 0.45 in CPs
    base_strings = []
    base_string = []
    # divide the base_string.
    base_set = np.arange(0,len(Sit0))
    while len(base_set) != 0:
        k = np.random.choice(base_set,1)[0]
        base_string +=[k]
        base_set = np.intersect1d(base_set,np.where(normalized_hamming_matrix[k]>Theta)[0])
    # find same strings.  < Delta
    for k in range(len(base_string)):
        base_strings += [list(np.where(normalized_hamming_matrix[base_string[k]] < Delta)[0])]
    
    # base_strings = sorted(base_strings, key=lambda x: len(x),reverse=True)
    
    return base_strings

def normalization(data):
    _range = np.max(data,0) - np.min(data,0)
    #np.max(,1) sum the column
    
    return (data - np.min(data,0)) / _range

def reconstruct_scatter(x,G,i):
    # reconstruct_scatter(x.value*(-1),G,i)
    # G[i] neighbours of node i.
    # x:x.value : -ln(1-lambda_i)a_ij
    # 创建数据
    Null = np.fromiter(set(np.arange(G.number_of_nodes()))-set(G[i])-{i},dtype=int)
    # x.value 对应 0,1,2,3,...i-1,i+1,...N-1 修正loc
    loc1 = np.where(np.array(G[i])>i,np.array(G[i])-1,np.array(G[i]))
    loc2 = np.where(Null>i,Null-1,Null)
    
    existent_links = np.random.rand(len(loc1), 2) / [5, 1] + [0.2, 0.2] #x(0.2~0.4)
    null_connections = np.random.rand(len(loc2), 2) / [5, 1] + [0.6, 0.6] #x(0.6~0.8)
    
    existent_links[:, 1] = x[loc1]
    null_connections[:, 1] = x[loc2]
    # 创建散点图.
    plt.figure(dpi=400)
    plt.scatter(existent_links[:, 0], existent_links[:, 1], color='green', label='Existent links')
    plt.scatter(null_connections[:, 0], null_connections[:, 1], color='orange', label='Null connections')
    plt.ylabel("$-ln(1-\lambda_i)a_{ij}$")
    plt.xlabel("$n_{\hat{t}}=$"+str(0.8)) #nt
    plt.xticks([])
    plt.xlim(0,1)
    plt.ylim(-0.2,0.6)
    plt.legend()
    plt.show()

