# CST_reconstruct temporal networks

#### 介绍
基于压缩感知理论，利用传播数据，进行网络重构


#### 内容说明

1.  `reconstruct dynamic network`：时序网络重构
2.  `reconstruct static network`：静态网络重构


#### 使用说明
    `reconstruct dynamic network`中实现了对生成动态网络（`ER`,`BA`,`WS`...）和真实动态网络的引入，实现了在`SIS`动力学和`CP`动力学下的传播与重构。
    可以在`main.py`文件中设置。
##### 1. 系统参数设置

```
n_nodes = 500 # 结点数目
p = 0.03  # ER:p BA:k=n_nodes*p
time_steps = 30 # 模拟传播和重构的时间步
num_simulations = 5000  # 独立传播次数
n0 = 30 # 初始感染人数
```
##### 2. 生成网络 or 真实网络？
生成网络
```
network = generate_networks.DynamicNetwork(n_nodes, p,'ER','Random') # Generative dynamic networks...#ER\BA\WS\ZK...
network.evolve(time_steps)
```
真实网络 需要在`generate_networks.Real_DynamicNetwork`手动检查、修改。
```
network = generate_networks.Real_DynamicNetwork('your file name.txt') # If you want reconstruct real dynamic networks...
```
##### 3. 传播
SIS动力学
```
model = SISModel(network,time_steps,num_simulations,n0=n0)  # SIS dynamics
```

CP动力学

```
model = CPModel(network,time_steps,num_simulations,n0=n0)  # CP dynamics
```

##### 4. 重构 t=t* 时的网络结构

```
choose_time = 3
G_reconstruct = reconstruction(model,choose_time)
```
文件保存

```
G = network.graph_history[choose_time]
nx.write_adjlist(G, "G.adjlist")
nx.write_adjlist(G_reconstruct, "G_reconstruct.adjlist")
```

##### 5. 绘图
- 拓扑对比图

```
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
```

- 分布对比图（度分布、聚类系数分布图、介数中心性分布图、接近中心性分布图）

```
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
```
##### 6. 准确性计算
    SRAC：总体成功率 <br> 
    SREL：重构存在边成功率 <br> 
    SRNC：重构空边存在率 <br> 

```
G_matrix = nx.to_numpy_matrix(G)
G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
print("SRAC:",SRAC(G_matrix,G_reconstruct_matrix))
print("SREL:",SREL(G_matrix,G_reconstruct_matrix))
print("SRNC:",SRNC(G_matrix,G_reconstruct_matrix))
print("CR:",CR(G_reconstruct_matrix))
```
##### 7. 重构连续时间的网络
`reconstruct dynamic network\read_excel`可读取绘图

```
results = pd.DataFrame(columns=['choose_time', '_SRAC', '_SREL', '_SRNC'])
reconstruct_times0 = time.time()
for choose_time in range(time_steps):
    print("reconstruct T=",choose_time)
    G_reconstruct = reconstruction(model,choose_time)
    G = network.graph_history[choose_time]
    G_matrix = nx.to_numpy_matrix(G)
    G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
    _SRAC = SRAC(G_matrix,G_reconstruct_matrix)
    _SREL = SREL(G_matrix,G_reconstruct_matrix)
    _SRNC = SRNC(G_matrix,G_reconstruct_matrix)
    results = results.append({'choose_time': choose_time, '_SRAC': _SRAC, '_SREL': _SREL, '_SRNC': _SRNC}, ignore_index=True)
    print("SRAC:",_SRAC)
    print("SREL:",_SREL)
    print("SRNC:",_SRNC)
reconstruct_times1 = time.time()
print("reconstruct simulated time(hr):",(reconstruct_times1-reconstruct_times0)/3600)
results.to_excel('results_SIS_BA.xlsx', index=False)
```

##### 8. 对于不同度的网络重构
运行`reconstruct dynamic network\main_for_k` <br>
`reconstruct dynamic network\read_excel`可读取绘图 <br>
参数含义与`main`相同，修改`p_all`得到平均度范围。
