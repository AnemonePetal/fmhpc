import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42	

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from sklearn.metrics import (
    roc_curve,
)
from sklearn import metrics
from util.Data import Data
from util.net_prep import get_feature_map
import seaborn as sns
import pandas as pd
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from util.Data import filter_df
from copy import deepcopy
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
from util.rename import rename_feat
import os
from functools import reduce

def plot_graph(
    args,
    graph,
    feature_map,
    save_path,
    with_legend=True,
    figsize=(22, 20),
    pos=None,
    colors=None,
    legend_labels_text=None,
    hide_nodes_flag=False,
    hide_isolated_flag=False,
    markers=None,
    legend_markers_text=None,
    communities_by = None,
    edgecolors = None,
    with_labels= False,
    show_legend=True,
):
    if communities_by != None and type(communities_by) == list:
        comm_nodes = [node for comm in communities_by for node in comm]
        graph = nx.subgraph(graph, comm_nodes)
    else:
        comm_nodes = list(range(len(feature_map)))
    if with_labels==False:
        figsize = (17,15)
    else:
        figsize = (17,15)
    figsize = (17,10)

    fig, ax = plt.subplots(figsize=figsize)
    if colors == None:
        cmap = plt.cm.plasma
        node_colors = cmap(np.linspace(0, 0.7, len(graph.nodes())))
        legend_node_colors = node_colors
    else:
        node_colors = colors
        legend_node_colors = list(dict.fromkeys(node_colors))
        
        if markers ==None:
            if len(legend_node_colors) == 2:
                nodeshapes_keys = ["o", "^"]
                color_marker_map = dict(zip(legend_node_colors, nodeshapes_keys))
                markers = [color_marker_map[color] for color in node_colors]
            elif len(legend_node_colors) == 3:
                nodeshapes_keys = ["o","s","^"]
            elif len(legend_node_colors) == 5:
                nodeshapes_keys = ["o","s","^","p","P"]
            else:
                raise ValueError("legend_node_colors should be 2 or 3 or 5")
        else:
            nodeshapes_keys = list(dict.fromkeys(markers))
    if legend_markers_text!=None:
        legend_markers_text = list(dict.fromkeys(legend_markers_text))

    if legend_labels_text != None:
        legend_labels_text = list(dict.fromkeys(legend_labels_text))
    else:
        node_labels = {i: label for i, label in enumerate(feature_map) if i in graph.nodes()}
        legend_labels_text = [str(k) + " " + rename_feat(v) for k, v in node_labels.items()]


    if pos == "spring":
        pos = nx.spring_layout(graph)
    elif pos == "fdp":
         pos = nx.nx_agraph.graphviz_layout(graph, prog="fdp")
    elif pos == "twopi":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="twopi")
    elif pos == "dot":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    elif pos == "sfdp":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="sfdp")
    elif pos == "circo":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="circo")
    elif pos == None or pos == "neato":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    elif pos == "circular":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="circo")
    elif pos == "kamada_kawai":
        pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    elif pos == "random":
        pos = nx.random_layout(graph)
    elif pos == "shell":
        pos = nx.shell_layout(graph)
    elif pos == "spectral":
        pos = nx.spectral_layout(graph)
    elif pos == "bipartite":
        pos = nx.bipartite_layout(graph, nodes=range(len(feature_map)))
    elif pos == "spiral":
        pos = nx.spiral_layout(graph)


    if colors == None and communities_by == None:
        nx.draw_networkx(graph,with_labels=True,width=1,node_size=700,node_color=node_colors,ax=ax,arrows=True,arrowstyle='->',pos=pos,font_color="whitesmoke")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        node_handles = [plt.plot([], [], marker="o", ls="", color=node_color)[0] for node_color in legend_node_colors]
        plt.legend(node_handles, legend_labels_text , loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 10})


    elif colors != None and communities_by == None:
        if len(legend_node_colors) == 2:
            group1 = [
                i for i, color in enumerate(node_colors) if color == legend_node_colors[0]
            ]
            group2 = [
                i for i, color in enumerate(node_colors) if color == legend_node_colors[1]
            ]
            visible_nodes = group1+group2

            if hide_isolated_flag:
                isolated_nodes = set(nx.isolates(graph))
                group1 = [i for i in group1 if i not in isolated_nodes]
                group2 = [i for i in group2 if i not in isolated_nodes]
                visible_nodes = group1+group2
            if args.model == 'gdn':
                pos[65] = (pos[65][0], pos[65][1]-30)
                pos[2] = (pos[2][0]-30, pos[2][1]-25)
                pos[51] = (pos[51][0]+20, pos[51][1]-10)
                pos[54] = (pos[54][0]-20, pos[54][1])
                pos[34] = (pos[34][0]-20, pos[34][1])
                pos[23] = (pos[23][0]-20, pos[23][1]+10)
                pos[29] = (pos[29][0]-30, pos[29][1])

            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=group1,
                node_shape=nodeshapes_keys[0],
                node_color=legend_node_colors[0],
                node_size=1800,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=group2,
                node_shape=nodeshapes_keys[1],
                node_color=legend_node_colors[1],
                node_size=1800,
                ax=ax,
            )

            nx.draw_networkx_edges(
                graph,
                pos,
                width=6,
                arrowsize=20,
                ax=ax,
                arrows=True,
                node_size=1800,
                arrowstyle="-|>",
            )

            if with_labels:
                node_labels = {}
                for node in graph.nodes():
                    node_labels[node] =node
                nx.draw_networkx_labels(
                    graph,
                    pos,
                    node_labels,
                    font_size=30,
                    font_color='w'
                )

            if len(legend_labels_text) == len(legend_node_colors):
                node_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker=nodeshapes_keys[i],
                        color="w",
                        label=legend_labels_text[i],
                        markerfacecolor=legend_node_colors[i],
                        markersize=30,
                    )
                    for i in range(len(legend_node_colors))
                ]
                if show_legend:
                    plt.legend(handles=node_handles, loc="upper center", ncol=len(node_handles), bbox_to_anchor=(0.5, 1.13),prop={'weight': 'bold', 'size': 30})

            else:
                node_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker=markers[i],
                        color="w",
                        label=legend_labels_text[i],
                        markerfacecolor=node_colors[i],
                        markersize=30,
                    )
                    for i in visible_nodes
                ]
                plt.subplots_adjust(right=0.5)
                leg = plt.legend(handles=node_handles, loc="center right", prop={'weight': 'bold', 'size': 28}, handlelength=0, handletextpad=0, bbox_to_anchor=(2.22, 0.5))
                for item in leg.legendHandles:
                    item.set_visible(False)


        else: 


            scale = 3
            pos[0] = (pos[0][0]*scale-185, pos[0][1]*scale-250)
            pos[1] = (pos[1][0]*scale-150, pos[1][1]*scale-250)
            pos[2] = (pos[2][0]*scale-140, pos[2][1]*scale-250)
            pos[3] = (pos[3][0]*scale-150, pos[3][1]*scale-250)
            pos[4] = (pos[4][0]*scale-170, pos[4][1]*scale-230)
            pos[5] = (pos[5][0]*scale-150, pos[5][1]*scale-250)
            pos[6] = (pos[6][0]*scale-180, pos[6][1]*scale-300)
            pos[7] = (pos[7][0]*scale-170, pos[7][1]*scale-240)
            pos[27] = (pos[27][0]+15, pos[27][1]+50)
            pos[26] = (pos[26][0], pos[26][1]+20)
            pos[14] = (pos[14][0], pos[14][1]-50)
            pos[12] = (pos[12][0], pos[12][1]-200)


            if hide_isolated_flag:
                isolated_nodes = set(nx.isolates(graph))
                visible_nodes = set(graph.nodes) -isolated_nodes

            for nodeshape in nodeshapes_keys:
                sub_nodes = [
                    node_idx for node_idx in comm_nodes if markers[node_idx] == nodeshape
                ]
                if sub_nodes == []: continue

                if hide_isolated_flag:
                    sub_nodes = [i for i in sub_nodes if i not in isolated_nodes]

                sub_node_colors = [node_colors[i] for i in sub_nodes]
                    
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist = sub_nodes,
                    node_shape=nodeshape,
                    node_color=sub_node_colors,
                    node_size=1800,
                    ax=ax,
                )

            if edgecolors != None:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    width=4,
                    arrowsize=15,
                    ax=ax,
                    arrows=True,
                    node_size=1800,
                    arrowstyle="-|>",
                    edge_color=edgecolors,
                    nodelist = list(comm_nodes)
                )
            else:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    width=4,
                    arrowsize=15,
                    ax=ax,
                    arrows=True,
                    node_size=1800,
                    arrowstyle="-|>",
                )

            if legend_labels_text != None and legend_markers_text !=None:
                node_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="s",
                        color="w",
                        label=legend_labels_text[i],
                        markerfacecolor=legend_node_colors[i],
                        markersize=25,
                    )
                    for i in range(len(legend_node_colors))
                ]
                marker_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker=nodeshapes_keys[i],
                        color="grey",
                        label=legend_markers_text[i],
                        markersize=25,
                    )
                    for i in range(len(nodeshapes_keys))
                ]
                handles = node_handles + marker_handles
                
                legend1 = plt.legend(handles=node_handles, loc="lower center", bbox_to_anchor=(0.5, -0.15), prop={'weight': 'bold',"size": 28}, ncol=4)
                legend2 = plt.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=len(marker_handles), prop={'weight': 'bold',"size": 28})

                plt.gca().add_artist(legend1)
                plt.gca().add_artist(legend2)

            else:
                node_handles = [
                    Line2D(
                        [0],
                        [0],
                        marker=markers[i],
                        color="w",
                        label=legend_labels_text[i],
                        markerfacecolor=node_colors[i],
                        markersize=30,
                    )
                    for i in visible_nodes
                ]
                plt.subplots_adjust(right=0.65)
                leg = plt.legend(handles=node_handles, loc="center right", prop={"size": 25}, handlelength=0, handletextpad=0, bbox_to_anchor=(1.6, 0.5))
                for item in leg.legendHandles:
                    item.set_visible(False)


    elif colors != None and communities_by == 'colors':
        communities = [set([i for i, color in enumerate(node_colors) if color == c]) for c in legend_node_colors]

        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        supergraph = nx.DiGraph()
        supergraph.add_nodes_from(range(len(communities)))
        for u, v in graph.edges():
            comm_u = node_to_community[u]
            comm_v = node_to_community[v]
            if comm_u != comm_v:  
                supergraph.add_edge(comm_u, comm_v)

        centers = list(nx.nx_agraph.graphviz_layout(supergraph, prog="dot").values())
        pos = {}
        for center, comm in zip(centers, communities):
            positions = nx.nx_agraph.graphviz_layout(nx.subgraph(graph, comm), prog="dot")
            scale = 0.5
            scaled_positions = {node: (x*scale, y*scale) for node, (x, y) in positions.items()}
            positions = {node: (center[0] + x_, center[1] + y_) for node, (x_, y_) in scaled_positions.items()}
            pos.update(positions)
        for sub_nodes, clr in zip(communities, legend_node_colors):
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist = sub_nodes,
                node_color= clr,
                node_size=500,
                ax=ax,
            )
        nx.draw_networkx_edges(
            graph,
            pos,
            width=6,
            arrowsize=15,
            ax=ax,
            arrows=True,
            node_size=500,
            arrowstyle="-|>",
        )
        node_handles = [
            Line2D(
                [0],
                [0],
                marker=".",
                color="w",
                label=legend_labels_text[i],
                markerfacecolor=legend_node_colors[i],
                markersize=25,
            )
            for i in range(len(legend_node_colors))
        ]
        legend1 = plt.legend(handles=node_handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(node_handles), prop={"size": 18})
    elif colors != None and type(communities_by) == list:
        communities = communities_by
        
        isolated_nodes = set(nx.isolates(graph))
        communities_connected = []
        communities_isolated = []
        for comm in communities:
            intersec = set(comm).intersection(isolated_nodes)
            if len(intersec) > 0 and len(intersec) < len(comm):
                intersec = set(comm).intersection(isolated_nodes)
                communities_connected.append(comm - intersec)
                communities_isolated.append(intersec)
            elif len(intersec)==0:
                communities_connected.append(comm)
            else:
                communities_isolated.append(intersec)

        communities_isolated = [*[set().union(*communities_isolated)]]
        communities = [*communities_connected, *communities_isolated]
        
        visible_nodes = reduce(lambda x, y: x.union(y), communities_connected)


        supergraph = extract_supergraph(graph, communities)
        centers = list(nx.nx_agraph.graphviz_layout(supergraph, prog="dot").values())
        
        super_communities = [*[set().union(*communities_connected)], *communities_isolated]
        supergraph_h = extract_supergraph(graph, super_communities)
        centers_h = list(nx.nx_agraph.graphviz_layout(supergraph_h, prog="neato").values())
        supergraph_l = extract_supergraph(graph, communities_connected)
        centers_l = list(nx.nx_agraph.graphviz_layout(supergraph_l, prog="dot").values())

        pos = {}
        for center, comm in zip(centers, communities):
            if comm in communities_isolated:
                positions = nx.nx_agraph.graphviz_layout(nx.subgraph(graph, comm), prog="neato")
                scale = 0.25
                scaled_positions = {node: (x*scale, y*scale) for node, (x, y) in positions.items()}
                positions = {node: (center[0]  +  x_ - 0.15, 0.4*center[1] + y_) for node, (x_, y_) in scaled_positions.items()}



                pos.update(positions)
            elif comm == {20,21}:
                positions = nx.nx_agraph.graphviz_layout(nx.subgraph(graph, comm), prog="dot")
                positions_values = sorted(positions.values(), key=lambda value: value[0])
                positions_keys = sorted(positions.keys())
                positions = {positions_keys[i]: positions_values[i] for i in range(len(positions_values))}
                scale = 0.3
                scaled_positions = {node: (x*scale, y*scale) for node, (x, y) in positions.items()}
                positions = {node: (center[0] + x_+11, center[1] + y_) for node, (x_, y_) in scaled_positions.items()}
                pos.update(positions)
            elif comm == {8,10,12}:
                positions = nx.nx_agraph.graphviz_layout(nx.subgraph(graph, comm), prog="dot")
                positions_values = sorted(positions.values(), key=lambda value: value[0])
                positions_keys = sorted(positions.keys())
                positions = {positions_keys[i]: positions_values[i] for i in range(len(positions_values))}
                scale = 0.3
                scaled_positions = {node: (x*scale, y*scale) for node, (x, y) in positions.items()}
                positions = {node: (center[0] + x_, center[1] + y_-15) for node, (x_, y_) in scaled_positions.items()}
                pos.update(positions)
            elif comm == {14,16,18}:
                positions = nx.nx_agraph.graphviz_layout(nx.subgraph(graph, comm), prog="dot")
                positions_values = sorted(positions.values(), key=lambda value: value[0])
                positions_keys = sorted(positions.keys())
                positions = {positions_keys[i]: positions_values[i] for i in range(len(positions_values))}
                scale = 0.3
                scaled_positions = {node: (x*scale, y*scale) for node, (x, y) in positions.items()}
                positions = {node: (center[0] + x_, center[1] + y_-15) for node, (x_, y_) in scaled_positions.items()}
                pos.update(positions)
            else:
                positions = nx.nx_agraph.graphviz_layout(nx.subgraph(graph, comm), prog="dot")
                positions_values = sorted(positions.values(), key=lambda value: value[0])
                positions_keys = sorted(positions.keys())
                positions = {positions_keys[i]: positions_values[i] for i in range(len(positions_values))}
                scale = 0.3
                scaled_positions = {node: (x*scale, y*scale) for node, (x, y) in positions.items()}
                positions = {node: (center[0] + x_, center[1] + y_) for node, (x_, y_) in scaled_positions.items()}
                pos.update(positions)
                


        for nodeshape in nodeshapes_keys:
            sub_nodes = [
                node_idx for node_idx in comm_nodes if markers[node_idx] == nodeshape
            ]
            if sub_nodes == []: continue
            if hide_isolated_flag:
                sub_nodes = [i for i in sub_nodes if i not in isolated_nodes]
            sub_node_colors = [node_colors[i] for i in sub_nodes]
                
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist = sub_nodes,
                node_shape=nodeshape,
                node_color=sub_node_colors,
                node_size=1800,
                ax=ax,
            )
        if with_labels:
            node_labels = {}
            for node in graph.nodes():
                node_labels[node] =node
            nx.draw_networkx_labels(
                graph,
                pos,
                node_labels,
                font_size=30,
                font_color='black'
            )

        if edgecolors != None:
            nx.draw_networkx_edges(
                graph,
                pos,
                width=4,
                arrowsize=15,
                ax=ax,
                arrows=True,
                node_size=1800,
                arrowstyle="-|>",
                edge_color=edgecolors,
                nodelist = list(comm_nodes)
            )
        else:
            nx.draw_networkx_edges(
                graph,
                pos,
                width=4,
                arrowsize=15,
                ax=ax,
                arrows=True,
                node_size=1800,
                arrowstyle="-|>",
            )


        if legend_labels_text != None and legend_markers_text !=None:
            node_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label=legend_labels_text[i],
                    markerfacecolor=legend_node_colors[i],
                    markersize=25,
                )
                for i in range(len(legend_node_colors))
            ]
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=nodeshapes_keys[i],
                    color="grey",
                    label=legend_markers_text[i],
                    markersize=25,
                )
                for i in range(len(nodeshapes_keys))
            ]

            if args.model == 'fm':
                legend1 = plt.legend(handles=node_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16), prop={'weight': 'bold', 'size': 28}, ncol=4)
                legend2 = plt.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=len(marker_handles), prop={'weight': 'bold', 'size': 28})
            else:
                legend1 = plt.legend(handles=node_handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(node_handles), prop={"size": 18})
                legend2 = plt.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=len(marker_handles), prop={"size": 18})
            plt.gca().add_artist(legend1)
            plt.gca().add_artist(legend2)

        else:
            node_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=markers[i],
                    color="w",
                    label=legend_labels_text[i],
                    markerfacecolor=node_colors[i],
                    markersize=30,
                )
                for i in visible_nodes
            ]
            plt.subplots_adjust(right=0.55)
            leg = plt.legend(handles=node_handles, loc="center right", prop={'weight': 'bold', 'size': 30}, handlelength=0, handletextpad=0, bbox_to_anchor=(1.8, 0.5))
            for item in leg.legendHandles:
                item.set_visible(False)



    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

def extract_supergraph(graph, communities):
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
    supergraph = nx.DiGraph()
    supergraph.add_nodes_from(range(len(communities)))
    for u, v in graph.edges():
        if u not in node_to_community or v not in node_to_community:
            continue
        comm_u = node_to_community[u]
        comm_v = node_to_community[v]
        if comm_u != comm_v:  
            supergraph.add_edge(comm_u, comm_v)
    return supergraph


def plot_loss(losses, save_path):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(save_path)
    plt.close()


def plot_deviation(df, keys, threshold, save_path):
    df = df.sort_values(by=["instance", "timestamp"])
    fig = go.Figure()

    partition_len = 60

    if len(df["instance"].unique()) >= partition_len:
        visible_bool_list = [False] * (partition_len * 2 + 1)
    else:
        visible_bool_list = [False] * (len(df["instance"].unique()) * 2 + 1)

    visible_bool_list[-1] = True
    button_list = []

    partition_id = 0
    for i, instance_name in enumerate(df["instance"].unique()):

        fig.add_scattergl(
            x=df[(df["instance"] == instance_name)]["timestamp"],
            y=df[df["instance"] == instance_name]["score"],
            line={"color": "black"},
            name=instance_name + " normal",
        )
        fig.add_trace(
            go.Scattergl(
                x=df[(df["instance"] == instance_name) & (df["pred"] == 1)][
                    "timestamp"
                ],
                y=df[(df["instance"] == instance_name) & (df["pred"] == 1)]["score"],
                mode="markers",
                marker=dict(color="red", size=8),
                name=instance_name + " abnormal",
            )
        )
        ii = i % partition_len
        visible_bool_list[ii * 2 : ii * 2 + 2] = [True] * 2
        button_list.append(
            dict(
                label=instance_name,
                method="update",
                args=[
                    {"visible": deepcopy(visible_bool_list)},
                    {"title": instance_name},
                ],
            )
        )
        visible_bool_list[ii * 2 : ii * 2 + 2] = [False] * 2

        if (i + 1) % partition_len == 0:
            fig.add_trace(
                go.Scattergl(
                    x=df["timestamp"].unique(),
                    y=[threshold] * len(df["timestamp"].unique()),
                    line={"color": "green", "dash": "dot"},
                    name="threshold",
                )
            )
            fig.update_layout(updatemenus=[dict(active=0, buttons=button_list)])
            fig.update_layout(
                xaxis_title="timestamp",
                yaxis_title="Graph deviation score",
            )
            fig.write_html(save_path[:-4] + "_partition" + str(partition_id) + ".html")
            partition_id += 1
            button_list = []
            fig = go.Figure()
            if len(df["instance"].unique()) < partition_len * (partition_id + 1):
                visible_bool_list = [False] * (
                    (len(df["instance"].unique()) % partition_len) * 2 + 1
                )
                visible_bool_list[-1] = True

    if (i + 1) % partition_len != 0:
        fig.add_trace(
            go.Scattergl(
                x=df["timestamp"].unique(),
                y=[threshold] * len(df["timestamp"].unique()),
                line={"color": "green", "dash": "dot"},
                name="threshold",
            )
        )
        fig.update_layout(updatemenus=[dict(active=0, buttons=button_list)])
        fig.update_layout(
            xaxis_title="timestamp",
            yaxis_title="Graph deviation score",
        )
        fig.write_html(save_path[:-5] + "_partition" + str(partition_id) + ".html")


def plot_deviation_all(df_list, keys, threshold, save_path):
    df_list = [df.sort_values(by=["instance", "timestamp"]) for df in df_list]
    fig = go.Figure()
    df_labels = ["train", "val", "test"]
    color_list = ["blue", "brown", "black"]

    partition_len = 10

    if len(df_list[2]["instance"].unique()) >= partition_len:
        visible_bool_list = [False] * (partition_len * 2 * 3 + 1)
    else:
        visible_bool_list = [False] * (len(df_list[2]["instance"].unique()) * 2 * 3 + 1)

    visible_bool_list[-1] = True
    button_list = []

    partition_id = 0
    for i, instance_name in enumerate(df_list[2]["instance"].unique()):
        timestamp_series = pd.concat(
            [df[(df["instance"] == instance_name)]["timestamp"] for df in df_list],
            ignore_index=True,
        )
        for j, df in enumerate(df_list):
            fig.add_scattergl(
                x=df[(df["instance"] == instance_name)]["timestamp"],
                y=df[df["instance"] == instance_name]["score"],
                line={"color": color_list[j]},
                name=df_labels[j] + ":" + instance_name + " normal",
            )
            fig.add_trace(
                go.Scattergl(
                    x=df[(df["instance"] == instance_name) & (df["pred"] == 1)][
                        "timestamp"
                    ],
                    y=df[(df["instance"] == instance_name) & (df["pred"] == 1)][
                        "score"
                    ],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name=df_labels[j] + ":" + instance_name + " abnormal",
                )
            )
        ii = i % partition_len
        visible_bool_list[ii * 6 : ii * 6 + 6] = [True] * 2 * 3
        button_list.append(
            dict(
                label=instance_name,
                method="update",
                args=[
                    {"visible": deepcopy(visible_bool_list)},
                    {"title": instance_name},
                ],
            )
        )
        visible_bool_list[ii * 6 : ii * 6 + 6] = [False] * 2 * 3

        if (i + 1) % partition_len == 0:
            fig.add_trace(
                go.Scattergl(
                    x=timestamp_series.unique(),
                    y=[threshold] * len(timestamp_series.unique()),
                    line={"color": "green", "dash": "dot"},
                    name="threshold",
                )
            )
            fig.update_layout(updatemenus=[dict(active=0, buttons=button_list)])
            fig.update_layout(
                xaxis_title="timestamp",
                yaxis_title="Graph deviation score",
            )
            fig.write_html(save_path[:-4] + "_partition" + str(partition_id) + ".html")
            partition_id += 1
            button_list = []
            fig = go.Figure()
            if len(df["instance"].unique()) < partition_len * (partition_id + 1):
                visible_bool_list = [False] * (
                    (len(df["instance"].unique()) % partition_len) * 2 * 3 + 1
                )
                visible_bool_list[-1] = True

    if (i + 1) % partition_len != 0:
        fig.add_trace(
            go.Scattergl(
                x=timestamp_series.unique(),
                y=[threshold] * len(timestamp_series.unique()),
                line={"color": "green", "dash": "dot"},
                name="threshold",
            )
        )
        fig.update_layout(updatemenus=[dict(active=0, buttons=button_list)])
        fig.update_layout(
            xaxis_title="timestamp",
            yaxis_title="Graph deviation score",
        )
        fig.write_html(save_path[:-5] + "_partition" + str(partition_id) + ".html")


def plot_deviation_all_onebyone(df_list, labels, threshold, save_path):
    df_list = [df.sort_values(by=["instance", "timestamp"]) for df in df_list]

    df_labels = ["train", "val", "test"]
    color_list = ["blue", "brown", "black"]

    if not os.path.exists(save_path):
        os.makedirs(save_path)



    for i, instance_name in enumerate(df_list[2]["instance"].unique()):
        fig = go.Figure()
        timestamp_series = pd.concat(
            [df[(df["instance"] == instance_name)]["timestamp"] for df in df_list],
            ignore_index=True,
        )
        for j, df in enumerate(df_list):
            fig.add_scattergl(
                x=df[(df["instance"] == instance_name)]["timestamp"],
                y=df[df["instance"] == instance_name]["score"],
                line={"color": color_list[j]},
                name=df_labels[j] + ":" + str(instance_name) + " normal",
                legendgroup=df_labels[j],
            )
            fig.add_trace(
                go.Scattergl(
                    x=df[(df["instance"] == instance_name) & (df["pred"] == 1)][
                        "timestamp"
                    ],
                    y=df[(df["instance"] == instance_name) & (df["pred"] == 1)][
                        "score"
                    ],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name=df_labels[j] + ":" + str(instance_name) + " abnormal pred",
                    legendgroup=df_labels[j],
                )
            )
            fig.add_trace(
                go.Scattergl(
                    x=df[(df["instance"] == instance_name) & (df["label"] == 1)][
                        "timestamp"
                    ],
                    y=df[(df["instance"] == instance_name) & (df["pred"] == 1)][
                        "score"
                    ],
                    mode="markers",
                    marker=dict(color="goldenrod", size=8),
                    name=df_labels[j] + ":" + str(instance_name) + " abnormal gt",
                    legendgroup=df_labels[j],
                )
            )

        fig.add_trace(
            go.Scattergl(
                x=timestamp_series.unique(),
                y=[threshold] * len(timestamp_series.unique()),
                line={"color": "green", "dash": "dot"},
                name="threshold",
            )
        )
        fig.update_layout(
            xaxis_title="timestamp",
            yaxis_title="Graph deviation score",
        )
        fig.write_html(save_path + "/" + str(instance_name).split(":")[0] + ".html")


def mse_plot(ori_gt, pred_df, feature_map=None, save_path="figs/msl/"):
    if feature_map == None:
        feature_map = get_feature_map("msl")
    colors = [
        "blue",
        "green",
        "yellow",
        "black",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
    ]
    labels = ["Normal", "Abnormal"]

    gt_df = ori_gt[ori_gt["timestamp"].isin(pred_df["timestamp"])]
    for i, feature in enumerate(feature_map):
        plt.figure(figsize=(20, 10))
        for j, label in enumerate(gt_df["label"].unique()):
            gt_df_slice = gt_df[gt_df["label"] == label]
            pred_df_slice = pred_df[pred_df["label"] == label]
            square_error = (gt_df_slice[feature] - pred_df_slice[feature]) ** 2
            mse = np.mean(square_error)
            plt.hist(
                square_error,
                bins=200,
                density=True,
                log=True,
                color=colors[j],
                alpha=0.5,
                label="{} (MSE = {:.4f})".format(labels[j], mse),
            )
        plt.title("MSE Distribution for {}".format(feature))
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        plt.savefig(save_path + "mse_{}.png".format(feature))
        plt.close()


def mse_plot_density(gt, pred, feature_map=None, save_path="figs/msl/"):
    if feature_map == None:
        feature_map = get_feature_map("msl")
    colors = [
        "blue",
        "green",
        "yellow",
        "black",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
    ]
    labels = ["Normal", "Abnormal"]
    gt = gt[gt["timestamp"].isin(pred["timestamp"])]
    for i, feature in enumerate(feature_map):
        plt.figure(figsize=(20, 10))
        for j, label in enumerate(gt["label"].unique()):
            gt_df_slice = gt[gt["label"] == label]
            pred_df_slice = pred[pred["timestamp"].isin(gt_df_slice["timestamp"])]
            square_error = (gt_df_slice[feature] - pred_df_slice[feature]) ** 2
            mse = np.mean(square_error)
            sns.kdeplot(
                data=square_error,
                alpha=0.5,
                log_scale=True,
                fill=True,
                common_norm=True,
                color=colors[j],
                bw_adjust=0.1,
                label="{} (MSE = {:.4f})".format(labels[j], mse),
            )
        plt.title("MSE Distribution for {}".format(feature))
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        plt.savefig(save_path + "mse_density_{}.png".format(feature))
        plt.close()


def mse_plot_density_all(train_gt, train_re, test_gt, test_re, mainenv, path):
    feature_map = get_feature_map("msl")
    colors = [
        "blue",
        "green",
        "purple",
        "yellow",
        "black",
        "orange",
        "pink",
        "brown",
        "gray",
    ]
    labels = ["Normal", "Abnormal"]

    test_gt = test_gt.iloc[mainenv.test_dataset.rang]
    train_gt = train_gt.iloc[mainenv.train_dataset.rang]

    test_gt = test_gt[test_gt["instance"].isin(test_re.instance.unique())]
    train_gt = train_gt[train_gt["instance"].isin(train_re.instance.unique())]

    test_gt.reset_index(drop=True, inplace=True)
    train_gt.reset_index(drop=True, inplace=True)

    if (test_gt["timestamp"] == test_re["timestamp"]).all() == False or test_gt.shape[
        0
    ] != test_re.shape[0]:
        raise ValueError("timestamp not match")
    if (
        train_gt["timestamp"] == train_re["timestamp"]
    ).all() == False or train_gt.shape[0] != train_re.shape[0]:
        raise ValueError("timestamp not match")


    for i, feature in enumerate(feature_map):
        plt.figure(figsize=(20, 10))
        for j, label in enumerate(range(1)):
            gt_df_slice = train_gt
            pred_df_slice = train_re
            square_error = (gt_df_slice[feature] - pred_df_slice[feature]) ** 2
            mse = np.mean(square_error)
            sns.kdeplot(
                data=square_error,
                alpha=0.5,
                log_scale=True,
                fill=True,
                common_norm=True,
                color=colors[j],
                bw_adjust=0.1,
                label="{} (MSE = {:.4f})".format("train_" + labels[j], mse),
            )

        for j, label in enumerate(test_gt["label"].unique()):
            gt_df_slice = test_gt[test_gt["label"] == label]
            pred_df_slice = test_re[test_re["label"] == label]
            square_error = (gt_df_slice[feature] - pred_df_slice[feature]) ** 2
            mse = np.mean(square_error)
            sns.kdeplot(
                data=square_error,
                alpha=0.5,
                log_scale=True,
                fill=True,
                common_norm=True,
                color=colors[j + 1],
                bw_adjust=0.1,
                label="{} (MSE = {:.4f})".format("test_" + labels[j], mse),
            )

        plt.title("MSE Distribution for {}".format(feature))
        plt.xlabel("MSE")
        plt.ylabel("Frequency")
        plt.legend(loc="upper left")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig("{}/{}.png".format(path, feature))
        plt.close()


def line_plot_2(ori_gt, pred_df, feature_map, savepath, data_filter=None):
    if data_filter != None:
        ori_gt = filter_df(ori_gt, data_filter)
        pred_df = filter_df(pred_df, data_filter)
    colors = [
        "blue",
        "green",
        "yellow",
        "black",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
    ]
    labels = ["Normal", "Abnormal"]
    gt_df = ori_gt[ori_gt["timestamp"].isin(pred_df["timestamp"])]
    gt_df = gt_df.reset_index(drop=True)
    if (pred_df["timestamp"] == gt_df["timestamp"]).all() == False:
        raise ValueError("timestamp not match")
    for i, feature in enumerate(feature_map):
        fig = go.Figure()
        gt_df_slice = gt_df
        pred_df_slice = pred_df
        r2 = metrics.r2_score(gt_df_slice[feature], pred_df_slice[feature])
        mse = np.mean((gt_df_slice[feature] - pred_df_slice[feature]) ** 2)
        fig.add_scattergl(
            x=gt_df_slice.mask(lambda x: x["label"] != 0)["timestamp"],
            y=gt_df_slice.mask(lambda x: x["label"] != 0)[feature],
            line={"color": "blue"},
            name="ground truth (normal)",
        )
        fig.add_scattergl(
            x=pred_df_slice.mask(lambda x: x["label"] != 0)["timestamp"],
            y=pred_df_slice.mask(lambda x: x["label"] != 0)[feature],
            line={"color": "deepskyblue"},
            name="prediction (normal)",
        )
        fig.add_scattergl(
            x=gt_df_slice.mask(lambda x: x["label"] != 1)["timestamp"],
            y=gt_df_slice.mask(lambda x: x["label"] != 1)[feature],
            line={"color": "red"},
            name="ground truth (abnormal)",
        )
        fig.add_scattergl(
            x=pred_df_slice.mask(lambda x: x["label"] != 1)["timestamp"],
            y=pred_df_slice.mask(lambda x: x["label"] != 1)[feature],
            line={"color": "bisque"},
            name="prediction (abnormal)",
        )
        fig.add_annotation(
            x=0.7,
            y=1.01,
            xref="paper",
            yref="paper",
            text="MSE = {:.3f}".format(mse),
            showarrow=False,
        )
        fig.add_annotation(
            x=0.3,
            y=1.01,
            xref="paper",
            yref="paper",
            text="R-squared = {:.3f}".format(r2),
            showarrow=False,
        )
        fig.update_layout(
            font=dict(size=20),
            title="{}".format(pred_df_slice["instance"].unique()[0]),
            xaxis_title="timestamp",
            yaxis_title="{}".format(feature),
            title_font=dict(size=30, family="Arial Black"),
            xaxis=dict(title_font_family="Arial Black", title_font=dict(size=30)),
            yaxis=dict(title_font_family="Arial Black", title_font=dict(size=30)),
            width=2000,
            height=2000 * 0.565,
        )
        if os.path.isdir(savepath) == False:
            Path(savepath).mkdir(parents=True, exist_ok=True)

        fig.write_html(savepath + "/line_{}.html".format(feature))



def line_plot_3(ori_gt, c_gt, pred_df, feature_map, savepath):

    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    c_gt["timestamp"] = pd.to_datetime(c_gt["timestamp"])

    colors = [
        "blue",
        "green",
        "yellow",
        "black",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
    ]
    labels = ["Normal", "Abnormal"]
    gt_df = ori_gt[ori_gt["timestamp"].isin(pred_df["timestamp"])]
    gt_df = gt_df.reset_index(drop=True)

    cgt_df = c_gt[c_gt["timestamp"].isin(pred_df["timestamp"])]
    cgt_df = cgt_df.reset_index(drop=True)
    if (pred_df["timestamp"] == gt_df["timestamp"]).all() == False:
        raise ValueError("timestamp not match")

    for i, feature in enumerate(feature_map):
        fig = go.Figure()
        gt_df_slice = gt_df
        pred_df_slice = pred_df
        r2 = metrics.r2_score(gt_df_slice[feature], pred_df_slice[feature])
        mse = np.mean((gt_df_slice[feature] - pred_df_slice[feature]) ** 2)
        fig.add_scattergl(
            x=gt_df_slice.mask(lambda x: x["label"] != 0)["timestamp"],
            y=gt_df_slice.mask(lambda x: x["label"] != 0)[feature],
            line={"color": "blue"},
            name="ground truth (normal)",
        )
        fig.add_scattergl(
            x=pred_df_slice.mask(lambda x: x["label"] != 0)["timestamp"],
            y=pred_df_slice.mask(lambda x: x["label"] != 0)[feature],
            line={"color": "deepskyblue"},
            name="prediction (normal)",
        )
        fig.add_scattergl(
            x=gt_df_slice.mask(lambda x: x["label"] != 1)["timestamp"],
            y=gt_df_slice.mask(lambda x: x["label"] != 1)[feature],
            line={"color": "red"},
            name="ground truth (abnormal)",
        )
        fig.add_scattergl(
            x=pred_df_slice.mask(lambda x: x["label"] != 1)["timestamp"],
            y=pred_df_slice.mask(lambda x: x["label"] != 1)[feature],
            line={"color": "bisque"},
            name="prediction (abnormal)",
        )

        fig.add_scattergl(
            x=cgt_df.mask(lambda x: x["label"] != 0)["timestamp"],
            y=cgt_df.mask(lambda x: x["label"] != 0)[feature],
            line={"color": "green"},
            name="corrupted ground truth (normal)",
        )
        fig.add_scattergl(
            x=cgt_df.mask(lambda x: x["label"] != 1)["timestamp"],
            y=cgt_df.mask(lambda x: x["label"] != 1)[feature],
            line={"color": "orange"},
            name="corrupted ground truth (abnormal)",
        )

        fig.update_layout(xaxis_title="timestamp", yaxis_title="{}".format(feature))
        fig.add_annotation(
            x=0.7,
            y=1.01,
            xref="paper",
            yref="paper",
            text="MSE = {:.3f}".format(mse),
            showarrow=False,
        )
        fig.add_annotation(
            x=0.3,
            y=1.01,
            xref="paper",
            yref="paper",
            text="R-squared = {:.3f}".format(r2),
            showarrow=False,
        )
        fig.write_html(savepath + "line_{}.html".format(feature))


def plot_roc(gt_labels, total_topk_err_scores, path, weight=None):
    if weight ==None:
        fpr, tpr, thresholds = roc_curve(gt_labels, total_topk_err_scores)
    else:
        fpr, tpr, thresholds = roc_curve(gt_labels, total_topk_err_scores, sample_weight=weight)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, "b", label="GDN (AUC = %0.4f)" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(path)
    plt.close()


def plot_pie(arr, path):
    freq_dict = dict(Counter(arr))
    labels = list(freq_dict.keys())
    values = list(freq_dict.values())

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_layout(title="Contributions of Features", font=dict(size=18))

    fig.write_html(path)


def plot_graph_wrapper(
    df_path, feature_list, node_filter=None, savepath=None, pos=None, colors=None
):
    if not os.path.exists(df_path):
        print("plot.py: Graph file not exist")
        return
    if savepath == None:
        savepath = df_path[:-4] + ".png"
    df_edges = pd.read_csv(df_path)
    df_edges = df_edges[["source", "destination"]]
    df_edges = df_edges[df_edges["source"] != df_edges["destination"]]
    if node_filter != None:
        node_filter_index = [feature_list.index(node) for node in node_filter]
        df_edges = df_edges[(~df_edges["destination"].isin(node_filter_index))]
        node_list_index = (
            df_edges["source"].unique().tolist()
            + df_edges["destination"].unique().tolist()
        )
        node_list_index = list(set(node_list_index))
        df_edges["source"] = df_edges["source"].apply(lambda x: feature_list[x])
        df_edges["destination"] = df_edges["destination"].apply(lambda x: feature_list[x])
        node_list = [feature_list[i] for i in node_list_index]
        df_edges["source"] = df_edges["source"].apply(lambda x: node_list.index(x))
        df_edges["destination"] = df_edges["destination"].apply(
            lambda x: node_list.index(x)
        )
        edge_list = df_edges.values.tolist()
    else:
        edge_list = df_edges.values.tolist()

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(node_list)))
    graph.add_edges_from(edge_list)
    plot_graph(graph, node_list, feature_list,save_path=savepath, pos=pos, colors=colors)


def plot_loss_wrapper(path):
    if not os.path.exists(path):
        print("plot.py: Loss file not exist")
        return
    path_noextension = path[:-4]
    train_log_epoch = pd.read_csv(path)
    plot_loss(train_log_epoch, save_path=path_noextension + ".png")


def plot_heatmap(gt_labels, total_topk_err_scores, threshold_list, path):
    gt_labels = np.array(gt_labels)
    labels_name = ["gt_normal", "gt_anomaly"]
    pred_labels_name = ["pred_normal", "pred_anomaly"]
    fig = make_subplots(rows=1, cols=1)
    for threshold in threshold_list:
        z = []
        for i, _ in enumerate(labels_name):
            gt_labels_mask = gt_labels == i
            scores_i = total_topk_err_scores[gt_labels_mask]
            z.append(
                [
                    np.sum(scores_i < threshold),
                    len(scores_i) - np.sum(scores_i < threshold),
                ]
            )
        z = np.array(z)
        trace = go.Heatmap(
            visible=False,
            z=z,
            x=pred_labels_name,
            y=labels_name,
            colorscale="Oranges",
        )
        fig.update_yaxes(autorange="reversed")
        fig.add_trace(trace)

    fig.data[0].visible = True

    steps = []
    for i, threshold in enumerate(threshold_list):
        step = dict(
            method="update",
            args=[{"visible": [s == i for s in range(10)]}],
            label="Threshold: " + str(threshold),
        )
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Threshold: "}, steps=steps)]

    fig.update_layout(sliders=sliders)

    fig.write_html(path, auto_open=True)
