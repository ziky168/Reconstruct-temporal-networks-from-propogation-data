# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:11:22 2024

@author: hp
For evaluate the reconstruction accuracy rate
"""

import numpy as np

def SRAC(original_adjacency, reconstructed_adjacency):
    """
    success rate for all connections,SRAC|计算模型预测正确的边的比例（准确性）
    Parameters:
    - original_adjacency (numpy.ndarray): 原始网络的邻接矩阵
    - reconstructed_adjacency (numpy.ndarray): 重构网络的邻接矩阵

    Returns:
    - accuracy (float): 模型预测正确的边的比例
    """
    # 确保输入的邻接矩阵是二维的numpy数组
    original_adjacency = np.array(original_adjacency)
    reconstructed_adjacency = np.array(reconstructed_adjacency)

    # 确保邻接矩阵的形状一致
    if original_adjacency.shape != reconstructed_adjacency.shape:
        raise ValueError("SRAC:原始网络和重构网络的邻接矩阵形状不一致")
    correct_predictions = np.sum(original_adjacency == reconstructed_adjacency)
    total_edges = original_adjacency.size
    accuracy = correct_predictions / total_edges
    return accuracy

def SREL(original_adjacency, reconstructed_adjacency):
    """
    success rate for existent links,SREL | 重构出存在边的成功率
    Parameters:
    - original_adjacency (numpy.ndarray): 原始网络的邻接矩阵
    - reconstructed_adjacency (numpy.ndarray): 重构网络的邻接矩阵
    
    Returns:
    - accuracy (float):  success rate for existent links
    """
    # 确保输入的邻接矩阵是二维的numpy数组
    original_adjacency = np.array(original_adjacency)
    reconstructed_adjacency = np.array(reconstructed_adjacency)
    if original_adjacency.shape != reconstructed_adjacency.shape:
        raise ValueError("SREL:原始网络和重构网络的邻接矩阵形状不一致")
    correct_predictions = len(np.intersect1d(np.where(original_adjacency.flatten()==1)[0],np.where(reconstructed_adjacency.flatten()==1)[0]))
    total_edges =  np.sum(original_adjacency==1)
    accuracy = correct_predictions / total_edges

    return accuracy


def SRNC(original_adjacency, reconstructed_adjacency):
    """
    success rate for null connections,SRNC | 重构出不存在边的成功率
    Parameters:
    - original_adjacency (numpy.ndarray): 原始网络的邻接矩阵
    - reconstructed_adjacency (numpy.ndarray): 重构网络的邻接矩阵
    
    Returns:
    - accuracy (float):  success rate for null connections
    """
    original_adjacency = np.array(original_adjacency)
    reconstructed_adjacency = np.array(reconstructed_adjacency)

    # 确保邻接矩阵的形状一致
    if original_adjacency.shape != reconstructed_adjacency.shape:
        raise ValueError("SRNC:原始网络和重构网络的邻接矩阵形状不一致")
    correct_predictions = len(np.intersect1d(np.where(original_adjacency.flatten()==0)[0],np.where(reconstructed_adjacency.flatten()==0)[0]))
    total_edges =  np.sum(original_adjacency==0)
    accuracy = correct_predictions / total_edges

    return accuracy

def CR(reconstructed_adjacency):
    """
    conflict rate,CR |冲突率
    Parameters:
    - reconstructed_adjacency (numpy.ndarray): 重构网络的邻接矩阵
    
    Returns:
    - conflict (float):  conflict rate
    """
    reconstructed_adjacency = np.array(reconstructed_adjacency)
    if len(reconstructed_adjacency.shape) != 2:
        raise ValueError("CR:重构网络邻接矩阵非二维")
    if reconstructed_adjacency.shape[0] != reconstructed_adjacency.shape[1]:
        raise ValueError("CR:重构网络非方阵")
    np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    Rtriu = np.triu(reconstructed_adjacency).T #上三角-转置
    Rtril = np.tril(reconstructed_adjacency) #下三角
    conflict = CR_tri(Rtriu, Rtril)
    
    return conflict

def CR_tri(original_adjacency, reconstructed_adjacency):
    """
    success rate for all connections,SRAC| 计算两个邻接矩阵下半部分的相似度
    Parameters:
    - original_adjacency (numpy.ndarray): 原始网络的邻接矩阵
    - reconstructed_adjacency (numpy.ndarray): 重构网络的邻接矩阵

    Returns:
    - accuracy (float): 模型预测正确的边的比例
    """
    # 确保输入的邻接矩阵是二维的numpy数组
    original_adjacency = np.array(original_adjacency)
    reconstructed_adjacency = np.array(reconstructed_adjacency)

    # 确保邻接矩阵的形状一致
    if original_adjacency.shape != reconstructed_adjacency.shape:
        raise ValueError("SRAC:原始网络和重构网络的邻接矩阵形状不一致")
    correct_predictions = np.sum(original_adjacency != reconstructed_adjacency)
    total_edges = original_adjacency.size/2
    accuracy = correct_predictions / total_edges
    return accuracy