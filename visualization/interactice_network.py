import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from util.net_prep import convert_adj2edges_wrapper
from util.net_wrapper import NetxGraph
from util.net_prep import get_feature_map
graph_path = 'results/jlab_fm_03-13--13-35-29/graph_datasize/12hour/profile_graph_adj.csv'
features = get_feature_map('data/jlab')
learned_graph_adj = pd.read_csv(graph_path)
len_nodes = learned_graph_adj.shape[0]
learned_graph = convert_adj2edges_wrapper(learned_graph_adj,len_nodes,dim=0, norm_flag=False)
df = pd.DataFrame(learned_graph,columns=['source','destination','value'])
df['source'] = df['source'].apply(lambda x: int(x))
df['destination'] = df['destination'].apply(lambda x: int(x))

graph= NetxGraph(df, features)
G = graph.graph
pos = nx.spring_layout(G, dim=3, seed=42)

Xn = [pos[k][0] for k in G.nodes()]
Yn = [pos[k][1] for k in G.nodes()]
Zn = [pos[k][2] for k in G.nodes()]

edge_x = []
edge_y = []
edge_z = []
edge_values = []
edge_text = []

for u, v in G.edges():
    x0, y0, z0 = pos[u]
    x1, y1, z1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])
    
    edge_value = G.get_edge_data(u, v)['weight']
    edge_text.extend([f'Value: {edge_value}', f'Value: {edge_value}', ''])
    edge_values.append(edge_value)

custom_colorscale = [
    [0.0, 'rgb(255,255,255)'],
    [0.2, 'rgb(230,230,230)'],
    [0.4, 'rgb(210,210,210)'],
    [0.7, 'rgb(180,180,180)'],
    [0.9, 'rgb(150,150,150)'],
    [1.0, 'rgb(125,125,125)']
]

log_edge_values = np.log1p(edge_values)

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(
        width=0.9, 
        color=log_edge_values,
        colorscale=custom_colorscale,
        colorbar=dict(
            thickness=15,
            title='Edge Values (log)',
            xanchor='left',
            titleside='right',
            x=1.0,
            y=0.8,
            len=0.4
        )
    ),
    hoverinfo='text',
    text=edge_text,
    mode='lines'
)

node_trace = go.Scatter3d(
    x=Xn, y=Yn, z=Zn,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right',
            x=1.0,
            y=0.3,
            len=0.4
        ),
        line_width=2))

node_adjacencies = []
node_text = []
for node in G.nodes():
    connections = len(list(G.neighbors(node)))
    node_adjacencies.append(connections)
    node_text.append(f'Node: {features[node]}<br>Connections: {connections}')

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Interactive 3D Network Graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    scene=dict(
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    )
                ))

fig.show()
fig.write_html("network_3d_visualization.html")
