# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:58:19 2024

@author: Rappy
"""

import numpy as np
import copy
# from numba import njit


# SIS传播模型  
class SISModel:  
    def __init__(self, network, time_steps, num_simulations,n0=10):  
        self.network = network
        self.N = self.network.n_nodes
        self.n0 = n0
        self.infection_probability = np.random.uniform(0.2,0.4,self.N)
        self.recovery_probability = np.random.uniform(0.4,0.6,self.N)
        self.time_steps = time_steps
        self.num_simulations = num_simulations
        self.node_states = np.zeros((self.num_simulations, self.time_steps+1, self.N), dtype=bool)
        # M,T,N

 
    def simulate(self):
        # # for each graph_history, simulating from different init states...
        self.init_states()
        for time_step, graph in enumerate(self.network.graph_history):
            if time_step < self.time_steps :
                # print(time_step)
                self.spreading_step(time_step, graph)
            else:
                return
            
        return
    
    # @njit(parallel=True)
    def init_states(self):
        # 确立每次传播的感染源
        LList = np.arange(0,self.N)
        for i in np.arange(self.num_simulations):
            series0 = np.random.choice(LList,self.n0,replace=False)
            self.node_states[i,0,series0] = True
    
    # @njit(parallel=True)
    def spreading_step(self,t,G):
        # 给定网络G,模拟t->t+1,每次传播过程。
        for simulation in range(self.num_simulations):
            # 感染过程
            infecting0 = set(np.where(self.node_states[simulation,t]==True)[0])
            infecting = self.propagation01(G,infecting0)
            self.node_states[simulation, t+1, list(infecting)] = True
            # series[:,:,times] = seriesonetime
            # 康复过程
            recovered = self.propagation10(infecting0)
            self.node_states[simulation, t+1, list(recovered)] = False
        return
    
    def propagation01(self,G,lastinfected_nodes):
        #从0到1,节点从未感染到感染过程
        neighbours = set() # 密接节点
        newinfected = set() # 新时刻感染者
        for inf in lastinfected_nodes:
            neighbours.update(set(G[inf])) # 密接节点
        neighbours = neighbours - lastinfected_nodes
        for node in list(neighbours):
            if self.prob01(G,lastinfected_nodes,node) > np.random.random(): #若 感染
                newinfected.add(node)
        newinfected.update(lastinfected_nodes)
        return newinfected
    
    def prob01(self,G,lastinfected_nodes,node):
        # 返回节点的感染概率Pn
        # neigh_infect = np.sum(np.isin(G[node], list(lastinfected_nodes)))  # 比下面这行要慢。
        neigh_infect = len(set(G[node]) & lastinfected_nodes) # 节点认识的人中有多少被感染的；
        Pn=1-pow(1-self.infection_probability[node],neigh_infect) # 节点 node 的感染概率
        return Pn
    
    def propagation10(self,lastinfected):
        #从1到0,康复过程;康复后可再次被感染;
        recovered = set()
        for node in list(lastinfected):
            if self.recovery_probability[node] > np.random.random():
                recovered.add(node)
        return recovered
    
# # SIS model over--------------------------

class CPModel:  
    def __init__(self, network, time_steps, num_simulations,n0=10):  
        self.network = network
        self.N = self.network.n_nodes
        self.n0 = n0
        self.infection_probability = np.random.uniform(0.7,0.9,self.N)
        self.recovery_probability = np.random.uniform(0.2,0.4,self.N)
        self.time_steps = time_steps
        self.num_simulations = num_simulations
        self.node_states = np.zeros((self.num_simulations, self.time_steps+1, self.N), dtype=bool)
        # M,T,N

 
    def simulate(self):
        # # for each graph_history, simulating from different init states...
        self.init_states()
        for time_step, graph in enumerate(self.network.graph_history):
            if time_step < self.time_steps :
                self.spreading_step(time_step, graph)
            else:
                return
        return
    
    # @njit(parallel=True)
    def init_states(self):
        # 确立每次传播的感染源
        LList = np.arange(0,self.N)
        for i in np.arange(self.num_simulations):
            series0 = np.random.choice(LList,self.n0,replace=False)
            self.node_states[i,0,series0] = True
    
    # @njit(parallel=True) # njit加速会有修改全局变量的问题。
    def spreading_step(self,t,G):
        # 给定网络G,模拟t->t+1,每次传播过程。
        for simulation in range(self.num_simulations):
            # 感染过程
            infecting0 = set(np.where(self.node_states[simulation,t]==True)[0])
            infecting = self.propagation01(G,infecting0)
            self.node_states[simulation, t+1, list(infecting)] = True
            # series[:,:,times] = seriesonetime
            # 康复过程
            recovered = self.propagation10(infecting0)
            self.node_states[simulation, t+1, list(recovered)] = False
        return
    
    def propagation01(self,G,lastinfected_nodes):
        #从0到1,节点从未感染到感染过程
        neighbours = set() # 密接节点
        newinfected = set() # 新时刻感染者
        for inf in lastinfected_nodes:
            neighbours.update(set(G[inf])) # 密接节点
        neighbours = neighbours - lastinfected_nodes
        for node in list(neighbours):
            if self.prob01(G,lastinfected_nodes,node) > np.random.random(): #若 感染
                newinfected.add(node)
        newinfected.update(lastinfected_nodes)
        return newinfected
    
    def prob01(self,G,lastinfected_nodes,node):
        # 返回节点的感染概率Pn
        # neigh_infect = np.sum(np.isin(G[node], list(lastinfected_nodes)))  # 比下面这行要慢。
        neigh_infect = len(set(G[node]) & lastinfected_nodes) # 节点认识的人中有多少被感染的；
        Pn = (self.infection_probability[node]/G.degree(node))*neigh_infect # 节点 node 的感染概率
        return Pn
    
    def propagation10(self,lastinfected):
        #从1到0,康复过程;康复后可再次被感染;
        recovered = set()
        for node in list(lastinfected):
            if self.recovery_probability[node] > np.random.random():
                recovered.add(node)
        return recovered

