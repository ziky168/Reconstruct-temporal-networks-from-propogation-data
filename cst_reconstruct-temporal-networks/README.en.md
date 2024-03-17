# CST_reconstruct temporal networks

#### Introduction
Based on compressive sensing theory, reconstruct network using propagation data(SIS/CP).


#### Content Description

1.  `reconstruct dynamic network`：Temporal network reconstruction
2.  `reconstruct static network`：Static network reconstruction


#### Usage
    In `reconstruct dynamic network`, it implements the introduction of generated dynamic networks (`ER`, `BA`, `WS`, etc.) and real dynamic networks, and realizes propagation and reconstruction under SIS dynamics and CP dynamics. <br>
    Settings can be configured in the main.py file.
##### 1. System Parameter Settings

```
n_nodes = 500 # Number of nodes
p = 0.03  # ER:p BA:k=n_nodes*p
time_steps = 30 # Time steps for simulation of propagation and reconstruction
num_simulations = 5000  # Number of independent propagation simulations
n0 = 30 # Initial number of infected individuals
```
##### 2. Generative Dynamic Network or Real Dynamic Network?
Generative Dynamic Network
```
network = generate_networks.DynamicNetwork(n_nodes, p,'ER','Random') # Generative dynamic networks...#ER\BA\WS\ZK...
network.evolve(time_steps)
```
Real Dynamic Network <br>
It needs to be manually checked and modified in `generate_networks.Real_DynamicNetwork`.

```
network = generate_networks.Real_DynamicNetwork('your file name.txt') # If you want reconstruct real dynamic networks...
```
##### 3. Propagation
SIS dynamics
```
model = SISModel(network,time_steps,num_simulations,n0=n0)  # SIS dynamics
```

CP dynamics

```
model = CPModel(network,time_steps,num_simulations,n0=n0)  # CP dynamics
```

##### 4. Reconstruct the network structure at t=t*

```
choose_time = 3
G_reconstruct = reconstruction(model,choose_time)
```
File saving

```
G = network.graph_history[choose_time]
nx.write_adjlist(G, "G.adjlist")
nx.write_adjlist(G_reconstruct, "G_reconstruct.adjlist")
```

##### 5. Plotting
- topology comparison graph

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

- Distribution comparison graph (degree distribution, clustering coefficient distribution, betweenness centrality distribution, closeness centrality distribution)

```
# Calculate degree distribution
degree_G0 = nx.degree(G)  
degree_G1 = nx.degree(G_reconstruct)  
  
# Calculate clustering coefficient distribution  
clustering_G0 = nx.clustering(G)  
clustering_G1 = nx.clustering(G_reconstruct)  
  
# Calculate betweenness centrality 
betweenness_G0 = nx.betweenness_centrality(G)  
betweenness_G1 = nx.betweenness_centrality(G_reconstruct)  
  
# Calculate closeness centrality
closeness_G0 = nx.closeness_centrality(G)  
closeness_G1 = nx.closeness_centrality(G_reconstruct)  

  
# Plot
plt.figure(figsize=(12, 12))  

plt.subplot(4, 1, 1)
degree_dist_G0 = [degree for node, degree in degree_G0]
degree_dist_G1 = [degree for node, degree in degree_G1]
plt.hist(degree_dist_G0, bins=np.arange(max(degree_dist_G0)+2)-0.5, alpha=0.5, color='b', label='G')
plt.hist(degree_dist_G1, bins=np.arange(max(degree_dist_G1)+2)-0.5, alpha=0.5, color='r', label='G_reconstruct')
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(4, 1, 2)
plt.hist(list(clustering_G0.values()), bins=20, alpha=0.5, color='b', label='G_origin')
plt.hist(list(clustering_G1.values()), bins=20, alpha=0.5, color='r', label='G_reconstruct')
plt.title('Clustering Coefficient Distribution')
plt.xlabel('Clustering Coefficient')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(4, 1, 3)
plt.hist(list(betweenness_G0.values()), bins=20, alpha=0.5, color='b', label='G_origin')
plt.hist(list(betweenness_G1.values()), bins=20, alpha=0.5, color='r', label='G_reconstruct')
plt.title('Betweenness Centrality Distribution')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')
plt.legend()

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
##### 6. Accuracy Calculation
    SRAC：Success Rate for All Connections <br> 
    SREL：Success Rate for Existing Links <br> 
    SRNC：Success Rate for None Connections <br> 

```
G_matrix = nx.to_numpy_matrix(G)
G_reconstruct_matrix = nx.to_numpy_matrix(G_reconstruct)
print("SRAC:",SRAC(G_matrix,G_reconstruct_matrix))
print("SREL:",SREL(G_matrix,G_reconstruct_matrix))
print("SRNC:",SRNC(G_matrix,G_reconstruct_matrix))
print("CR:",CR(G_reconstruct_matrix))
```
##### 7. Reconstruct continuous-time networks
You can read`reconstruct dynamic network\read_excel` for plotting.

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

##### 8. Network Reconstruction for Different Degrees
Run `reconstruct dynamic network\main_for_k` <br>
You can read `reconstruct dynamic network\read_excel`for plotting. <br>
ame parameters as `main`, modify the `p_all` to get the average degree range.
