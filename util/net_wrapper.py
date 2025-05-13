import os
import networkx as nx
import pandas as pd
import numpy as np
import math
from util import net_backbone

class NetxGraph:
    def __init__(self, df_path, node_list, node_filter=None):
        self.graph = nx.DiGraph()
        self.node_list = []
        self.edge_list = []
        self.weight_list = []
        self.read_graph(df_path, node_list, node_filter=node_filter)
    
    def read_graph(self, df_path, node_list, node_filter=None):
        if type(df_path) == str:
            if not os.path.exists(df_path):
                print("plot.py: Graph file not exist")
                return
            df_edges = pd.read_csv(df_path)
        elif type(df_path) == pd.DataFrame:
            df_edges = df_path
        df_edges = df_edges[df_edges["source"] != df_edges["destination"]]
        if node_filter != None:
            node_filter_index = [node_list.index(node) for node in node_filter]
            df_edges = df_edges[(~df_edges["destination"].isin(node_filter_index))]
            node_list_index = (
                df_edges["source"].unique().tolist()
                + df_edges["destination"].unique().tolist()
            )
            node_list_index = list(set(node_list_index))
            df_edges["source"] = df_edges["source"].apply(lambda x: node_list[x])
            df_edges["destination"] = df_edges["destination"].apply(lambda x: node_list[x])
            node_list = [node_list[i] for i in node_list_index]
            df_edges["source"] = df_edges["source"].apply(lambda x: node_list.index(x))
            df_edges["destination"] = df_edges["destination"].apply(
                lambda x: node_list.index(x)
            )
            edge_list = df_edges[["source", "destination"]].values.tolist()
            weight_list = df_edges["value"].values.tolist()
        else:
            edge_list = df_edges[["source", "destination"]].values.tolist()
            weight_list = df_edges["value"].values.tolist()

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(node_list)))
        for _, row in df_edges.iterrows():
            graph.add_edge(int(row["source"]), int(row["destination"]), weight=row["value"])        
        self.graph = graph
        self.node_list = node_list
        self.edge_list = edge_list
        self.weight_list = weight_list

    def get_communities(self):
         community = nx.algorithms.community.louvain_communities(self.graph)
         community = nx.algorithms.community.greedy_modularity_communities(self.graph, weight= 'weight')
         return
    
    def get_summary(self):
        summary = nx.snap_aggregation(self.graph)



def reduction_spectral(A,edges=10, epsilon=0.01):
    W = np.copy(A[np.nonzero(A)]).flatten()
    G = nx.DiGraph(A)

    laplacian = nx.linalg.laplacianmatrix.directed_laplacian_matrix(G)
    
    incidence = nx.linalg.graphmatrix.incidence_matrix(G,oriented=True)

    epsilon = epsilon
    k = round(24*math.log2(G.number_of_nodes()/(epsilon**2)))
    m = G.number_of_edges()
    choices = [1/k**0.5,-1/k**0.5]
    Q = np.random.choice(choices, (k,m))

    W_mat = np.diag(W)
    incidence_np = incidence.todense()
    Y = np.matmul(Q,np.sqrt(W_mat))
    Y = np.matmul(Y,incidence_np.transpose())
    Z = np.linalg.lstsq(laplacian,Y.transpose())
    Z_t = Z[0].copy()
    Re = np.zeros(G.number_of_edges())
    counter = 0
    for start,end in G.edges():
        temp = np.copy((Z_t[start] - Z_t[end])).flatten()
        Re[counter] = np.linalg.norm(temp,ord=2,axis=None,keepdims=False)
        counter += 1
    Re_norm = Re*W
    Re_norm = Re_norm/Re_norm.sum()
    q = edges
    Re_weight = W/(Re_norm*q)
    choices = list(range(G.number_of_edges()))
    edgesNew = np.random.choice(choices, q, p = Re_norm)
    H = nx.DiGraph()
    for _ in range(G.number_of_nodes()):
        H.add_node(_)
    for edge in edgesNew:
        start, end = list(G.edges())[edge]
        if not H.has_edge(start,end): H.add_edge(start,end, weight=Re_weight[edge])
        else: H[start][end]["weight"] += Re_weight[edge]
    return nx.adjacency_matrix(H).todense()

def reduction(G,type='disparity_filter',edges=100):
    table,_,_ = net_backbone.from_nx(G)
    if type == 'disparity_filter':
        table_red = net_backbone.disparity_filter(table)
        new_table = net_backbone.thresholding_edges(table_red, edges)
    elif type == 'noise_corrected':
        table_red = net_backbone.noise_corrected(table)
        new_table = net_backbone.thresholding_edges(table_red, edges)
    elif type == 'high_salience_skeleton':
        table_red = net_backbone.high_salience_skeleton(table)
        new_table = net_backbone.thresholding_edges(table_red, edges)
    elif type == 'weight':
        table_red = net_backbone.naive(table)
        new_table = net_backbone.thresholding_edges(table_red, edges)
    elif type == 'maximum_spanning_tree':
        new_G = net_backbone.maximum_spanning_tree(table)
        return new_G
    elif type == 'maximum_Edmonds_tree':
        new_G = net_backbone.maximum_Edmonds_tree(table)
        return new_G
    new_G= nx.from_pandas_edgelist(new_table, 'src', 'trg', ['nij'],create_using=nx.DiGraph())
    return new_G

def find_subgraph_with_hops(graph, node, hops):
    neighbors = set()
    current_level = {node}
    for _ in range(hops):
        next_level = set()
        for n in current_level:
            next_level.update(set(graph.predecessors(n)))
        neighbors.update(next_level)
        current_level = next_level
    
    neighbors.add(node)
    
    subgraph = graph.subgraph(neighbors)
    return subgraph
