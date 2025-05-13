import glob
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import os
from torch_geometric.utils import remove_self_loops, add_self_loops

class graph_reader:
    def __init__(self, path, args):
        if not os.path.isfile(path):
            self.edges = []
            self.node_dict = {node: i for i, node in enumerate(args.hnodes)}
        else:
            self.graph = self.hierarchical_graph_reader(path)
            self.args = args
            self._create_node_indices()
            self._setup_graph_h1()

    def hierarchical_graph_reader(self, path):
        """
        Reading the macro-level graph from disk.
        :param path: Path to the edge list.
        :return graph: Hierarchical graph as a NetworkX object.
        """
        edges = pd.read_csv(path).values.tolist()
        graph = nx.from_edgelist(edges)
        return graph

    def _setup_graph_h1(self):
        self.edges = [[edge[0], edge[1]] for edge in self.graph.edges()]
        self.edges = self.edges + [[edge[1], edge[0]] for edge in self.graph.edges()]
        self.node_dict = {node: i for i, node in enumerate(self.args.hnodes)}

        self.edges = [
            edge
            for edge in self.edges
            if edge[0] in self.node_dict and edge[1] in self.node_dict
        ]

        self.edges = torch.t(
            torch.LongTensor(
                [
                    [self.node_dict[edge[0]], self.node_dict[edge[1]]]
                    for edge in self.edges
                    if edge[0] in self.node_dict and edge[1] in self.node_dict
                ]
            )
        )

    def _create_node_indices(self):
        self.node_indices = [index for index in range(self.graph.number_of_nodes())]
        self.node_indices = torch.LongTensor(self.node_indices)


def store_featurelist(dataset, df):
    if os.path.isfile(f"{dataset}/list.txt"):
        raise Exception(f"{dataset}/list.txt already exists")
    feature_list = extract_featurelist(df)
    with open(f"{dataset}/list.txt", "w") as f:
        for item in feature_list:
            f.write("%s\n" % item)
    return feature_list

def extract_featurelist(df):
    feature_list = []
    for i in df.columns:
        excluded_values = [
            "attack",
            "timestamp",
            "label",
            "timestamp_memory",
            "timestamp_disk",
            "timestamp_slurm",
            "instance",
            "timestamp_network",
            "timestamp_cpu",
            "timestamp_gpu",
            "timestamp_power",
            "timestamp_job",
            "timestamp_job_event",
            "timestamp_job_resrc_usage",
            "timestamp_job_resrc_used",
            "timestamp_job_resrc_alloc",
            "node",
            "node_state",
            "hostname",
        ]
        if i not in excluded_values:
            feature_list.append(i)
    return feature_list


def get_feature_map(dataset):
    if not os.path.isfile(f"{dataset}/list.txt"):
        print(f"{dataset}/list.txt not found")
        return None
    else:
        feature_file = open(f"{dataset}/list.txt", "r")
        feature_list = []
        for ft in feature_file:
            if "#" not in ft:
                feature_list.append(ft.strip())
        return feature_list


def get_fc_graph_struc(dataset):
    feature_file = open(f"{dataset}/list.txt", "r")

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map



def check_selfloops(edges,features):
    edges, _ = remove_self_loops(edges)
    edges, _ = add_self_loops(edges, num_nodes=len(features))
    return edges


@torch.no_grad()
def convert_adj2edges_topk(graph, topk_num, dim=1, quantile=0):
    if topk_num == -1:
        topk_num = graph.size(0)

    graph = graph.clone()
    if quantile > 0:
        graph[torch.abs(graph) < torch.quantile(torch.abs(graph), quantile)] = 0

    topk_values, topk_indices = torch.topk(torch.abs(graph), topk_num, dim=dim)
    topk_values = graph.gather(dim, topk_indices)
    new_matrix = torch.zeros_like(graph)
    if dim == 1 or dim == -1:
        for i in range(graph.size(0)):
            new_matrix[i, topk_indices[i]] = topk_values[i]
    elif dim == 0:
        for i in range(graph.size(1)):
            new_matrix[topk_indices[:, i], i] = topk_values[:, i]
    edges = torch.nonzero(new_matrix).t()
    edge_values = new_matrix[edges[0], edges[1]]
    edges = torch.stack((edges[0], edges[1], edge_values))
    return edges
    
def convert_adj2edges_topk_silly(graph, topk_num, dim=1, quantile=0):
    if topk_num == -1:
        topk_num = graph.shape[0]
    graph = graph.clone()
    if quantile > 0:
        graph[torch.abs(graph) < torch.quantile(torch.abs(graph), quantile)] = 0
    if dim == 1:
        pass
    return 

def convert_adj2edges(adj,filter_zero=False):
    if type(adj)==pd.DataFrame:
        adj = adj.to_numpy()
    if adj.shape[0] != adj.shape[1] or adj.ndim != 2:
        raise Exception("adj should be square matrix")
    rows, cols = np.indices((adj.shape[0], adj.shape[0]))
    edges = np.vstack([rows.ravel(), cols.ravel(), adj.ravel()])
    if filter_zero:
        edges = edges[:, edges[2, :] > 0]
    return edges.T

def convert_edges2adj(edges, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    if isinstance(edges, torch.Tensor):
        edges = edges.numpy()
    for edge in edges:
        adj[int(edge[0]), int(edge[1])] = edge[2]
    return adj    

def check_graph_sparse(adj,topk,dim):
    non_zero_adj = torch.nonzero(adj)
    if dim == 1:
        for i in range(adj.size(0)):
            if len(non_zero_adj[non_zero_adj[:,0]==i]) != topk:
                print(f"node {i} has {len(non_zero_adj[non_zero_adj[:,0]==i])} edges")
    elif dim == 0:
        for i in range(adj.size(1)):
            if len(non_zero_adj[non_zero_adj[:,1]==i]) != topk:
                print(f"node {i} has {len(non_zero_adj[non_zero_adj[:,1]==i])} edges")
    return True

def norm_graph_adj(graph):
    graph_without_diag = graph - torch.diag(torch.diag(graph))
    graph = graph / torch.sum(abs(graph),1)
    graph = graph / torch.sum(abs(graph),0)
    return graph

def convert_adj2edges_wrapper(graph, nodes_num, topk=-1, dim=-1, quantile=0, norm_flag = False):
    if isinstance(graph, torch.Tensor):
        graph = graph.clone()
        graph = graph.to('cpu')
    elif isinstance(graph, np.ndarray):
        graph = graph.copy()
        graph = torch.from_numpy(graph).to('cpu')
    elif isinstance(graph, pd.DataFrame):
        graph = graph.copy()
        graph = torch.from_numpy(graph.values).to('cpu')
    else:
        raise Exception("graph type should be tensor, numpy, dataframe")
    
    if graph.ndim != 2:
        raise Exception("graphs should be 2d array")
    if graph.shape[0]>graph.shape[1]:
        graph = graph.T
    if graph.shape[0] == graph.shape[1] and graph.shape[0] == nodes_num:
        if norm_flag: graph = norm_graph_adj(graph)
        if dim!= -1:
            edges = convert_adj2edges_topk(graph, topk, dim=dim, quantile=quantile)
        else:
            edges_dim0 = convert_adj2edges_topk(graph, topk, dim=0, quantile=quantile)
            edges_dim1 = convert_adj2edges_topk(graph, topk, dim=1, quantile=quantile)
            edges = torch.cat((edges_dim0, edges_dim1), dim=1)
            edge_dict = {}
            for i in range(edges.shape[1]):
                start, end, weight = edges[:, i].tolist()
                if (start, end) not in edge_dict:
                    edge_dict[(start, end)] = weight
            merged_edges = torch.tensor([[start, end, weight] for (start, end), weight in edge_dict.items()]).T

    elif graph.shape[0]==3:
        graph_adj = convert_edges2adj(graph.T, nodes_num)
        graph_adj = torch.Tensor(graph_adj)
        if dim!= -1:
            edges = convert_adj2edges_topk(graph_adj, topk, dim=dim, quantile=quantile)
        else:
            edges_dim0 = convert_adj2edges_topk(graph_adj, topk, dim=0, quantile=quantile)
            edges_dim1 = convert_adj2edges_topk(graph_adj, topk, dim=1, quantile=quantile)
            edges = torch.cat((edges_dim0, edges_dim1), dim=1)
            edge_dict = {}
            for i in range(edges.shape[1]):
                start, end, weight = edges[:, i].tolist()
                if (start, end) not in edge_dict:
                    edge_dict[(start, end)] = weight
            merged_edges = torch.tensor([[start, end, weight] for (start, end), weight in edge_dict.items()]).T

    elif graph.shape[0]==2:
        graph = torch.cat((graph, torch.ones((1,graph.shape[1]))), axis=0)
        edges = graph
    return edges.T


if __name__ == "__main__":
    graph_reader()
