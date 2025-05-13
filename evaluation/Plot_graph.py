import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import pandas as pd
import numpy as np
from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from util.net_prep import convert_adj2edges_wrapper
from util.plot import plot_graph
from util.save import save_graph
import matplotlib.pyplot as plt
from util.net_wrapper import NetxGraph
from util.net_prep import get_feature_map, convert_edges2adj
import networkx as nx

def find_constant_columns(data):
    constant_columns = []
    for column in data.columns:
        if len(data[column].unique()) == 1:
            constant_columns.append(column)
    return constant_columns

def column2id(columns, features):
    columns_id = []
    for column in columns:
        columns_id.append(features.index(column))
    return columns_id

def reload_almost_constant_columns(datapath='./data/jlab'):
    if not os.path.exists(datapath+'/cons_columns.txt') or not os.path.exists(datapath+'/acons_columns.txt'):
        return None, None
    with open(datapath+'/cons_columns.txt', 'r') as f:
        cons_columns = f.readlines()
    cons_columns = [x.strip() for x in cons_columns]
    with open(datapath+'/acons_columns.txt', 'r') as f:
        acons_columns = f.readlines()
    acons_columns = [x.strip() for x in acons_columns]
    return cons_columns, acons_columns

def find_almost_constant_columns(df,features, cache=True, datapath='./data/jlab'):
    cons_feats = []
    acons_feats = []
    df_nunique = df.nunique()
    for feature in features:
        if df_nunique[feature]==1:
            cons_feats.append(feature)

    df_nunique = df.groupby('instance').agg(['nunique']).reset_index()
    df_nunique.columns = [col[0] for col in df_nunique.columns.values]

    df_1_nunique_percent = (df_nunique==1).sum()/(df_nunique.shape[0])
    df_1_nunique_percent = df_1_nunique_percent[features].to_frame(name='cons_per').reset_index(names='feature')
    acons_feats = df_1_nunique_percent[(df_1_nunique_percent['cons_per']==1)].feature.values.tolist()
    acons_feats = list(set(acons_feats)-set(cons_feats))
    
    cons_ucons_feats = df_1_nunique_percent[(df_1_nunique_percent['cons_per']>0) &  (df_1_nunique_percent['cons_per']<1)].feature.values.tolist()
    unconstant_instances = ['']*len(features)
    df_1_nunique_percent['unconstant_instances'] = unconstant_instances
    cons_ucons_feats_mask = df_1_nunique_percent['feature'].isin(cons_ucons_feats)
    max_instance_values = {}
    for col in cons_ucons_feats:
            max_index = df_nunique[col].idxmax()
            max_instance_value = df_nunique.loc[max_index, 'instance']
            max_instance_values[col] = max_instance_value
    df_1_nunique_percent.loc[cons_ucons_feats_mask,'unconstant_instances']=list(max_instance_values.values())

    if cache:
        with open(datapath+'/cons_columns.txt', 'w') as f:
            for item in cons_feats:
                f.write("%s\n" % item)
        with open(datapath+'/acons_columns.txt', 'w') as f:
            for item in acons_feats:
                f.write("%s\n" % item)
        df_1_nunique_percent.to_csv(datapath+'/features_cons_percent.csv')
    return cons_feats, acons_feats

def olcfcutsec_colors(features):
    new_features = []
    for feature in features:
        if feature.startswith('gpu'):
            feature = feature.replace('gpu0', 'p0_gpu0').replace('gpu1', 'p0_gpu1').replace('gpu2', 'p0_gpu2').replace('gpu3', 'p1_gpu0').replace('gpu4', 'p1_gpu1').replace('gpu5', 'p1_gpu2')
        new_features.append(feature)
    features = new_features

    cmap = plt.cm.tab10
    colors = []
    legend_labels_text = []
    markers = []
    legend_markers_text = []

    for feature in features:
        if feature.startswith('p0_gpu0'):
            legend_labels_text.append('GPU0')
            colors.append('#006BA4')
        elif feature.startswith('p0_gpu1'):
            legend_labels_text.append('GPU1')
            colors.append('#5F9ED1')
        elif feature.startswith('p0_gpu2'):
            legend_labels_text.append('GPU2')
            colors.append('#A2C8EC')
        elif feature.startswith('p1_gpu0'):
            legend_labels_text.append('GPU3')
            colors.append('#FF800E')
        elif feature.startswith('p1_gpu1'):
            legend_labels_text.append('GPU4')
            colors.append('#C85200')
        elif feature.startswith('p1_gpu2'):
            legend_labels_text.append('GPU5')
            colors.append('#FFBC79')
        elif feature.startswith('p0_temp') or feature.startswith('p1_temp') or feature.startswith('p0_power') or feature.startswith('p1_power'):
            legend_labels_text.append('CPU')
            colors.append('#595959')
        elif feature.startswith('ps0_input') or feature.startswith('ps1_input'):
            legend_labels_text.append('WholePower')
            colors.append('#898989')
        else:
            legend_labels_text.append('Others')
            colors.append('#898989')

        if 'temp' in feature:
            markers.append('o')
            legend_markers_text.append('Temperature')
        if 'power' in feature:
            markers.append('s')
            legend_markers_text.append('Power')
    return colors, legend_labels_text, markers, legend_markers_text

def nei_both(graph,i=1):
    nei_graph_in_edges = graph.in_edges(i,data='weight')
    nei_graph_out_edges = graph.out_edges(i,data='weight')
    nei_graph_in = nx.DiGraph()
    nei_graph_in.add_weighted_edges_from(nei_graph_in_edges)
    nei_graph_out = nx.DiGraph()
    nei_graph_out.add_weighted_edges_from(nei_graph_out_edges)
    return nei_graph_in,nei_graph_out

def count_edges(G, comm0, comm1):
    if comm0 == comm1:
        return len([e for e in G.edges() if e[0] in comm0 and e[1] in comm0])
    else:
        return len([e for e in G.edges() if (e[0] in comm0 and e[1] in comm1) or (e[0] in comm1 and e[1] in comm0)])

def calculate_comm_relation_matrix(graph,communities):
    communities_temp = [communities[0].union(communities[1]), communities[2].union(communities[3]), communities[4], communities[5]]
    class_names = ['GPU 0,1,2',  'GPU 3,4,5', 'CPU', 'WholePower']
    num_conn_comm = []
    g_temp = graph.copy()
    
    result_matrix = []
    for i in range(len(communities_temp)):
        result_matrix.append([])
        for j in range(len(communities_temp)):
            result_matrix[i].append(count_edges(g_temp, communities_temp[i], communities_temp[j]))
    df = pd.DataFrame(result_matrix, index=class_names, columns=class_names)
    print(df.to_markdown())


def prepare_draw(args):
    communities = None
    if args.dataset == 'jlab':
        cons_columns, acons_columns = reload_almost_constant_columns(datapath=args.paths['dataset'])
        if cons_columns is None or acons_columns is None:
            data = Dataset(args)
            cons_columns, acons_columns = find_almost_constant_columns(data.train, args.features,datapath=args.paths['dataset'])
        colors =[]
        legend_labels_text = []
        nodeshapes = None
        legend_markers_text = None
        for i in range(len(args.features)):
            if args.features[i] in cons_columns:
                colors.append('#FF800E')
                legend_labels_text.append('Constant')
            else:
                colors.append('#006BA4')
                legend_labels_text.append('Non-constant')

    elif args.dataset == 'olcfcutsec':
        colors, legend_labels_text, nodeshapes, legend_markers_text = olcfcutsec_colors(args.features)
        nodeshapes_label = {'o':'Temp', 's':'Power'}
        detailed_labels = [ x + nodeshapes_label[y] for x,y in zip(legend_labels_text, nodeshapes)]
        communities = [('GPU0Temp', 'GPU1Temp', 'GPU2Temp'), ('WholePowerPower'), ('GPU3Temp', 'GPU4Temp', 'GPU5Temp'), ('GPU0Power', 'GPU1Power', 'GPU2Power'), ('GPU3Power', 'GPU4Power', 'GPU5Power'), ('CPUTemp', 'CPUPower')]
        communities = [set([i for i, legend_label in enumerate(detailed_labels) if legend_label in c]) for c in communities]
    return communities,colors,legend_labels_text,nodeshapes,legend_markers_text

def draw(args, dataset, model, graph_path, pos, local_top, global_top, local_top_direction=-1,detail_flag=False, savepath='./fig.png'):
    args.dataset=dataset
    args.model = model
    prepare_env(args)
    args.features = get_feature_map(args.paths['dataset'])
    communities, colors, legend_labels_text, nodeshapes, legend_markers_text = prepare_draw(args)
    args.topk = local_top
    args.paths['graph_csv'] = graph_path
    hide_nodes_flag= False
    if args.model == 'fm':
        args.paths['profile_graph_csv'] = os.path.dirname(args.paths['graph_csv']) + '/profile_graph.csv'
        args.paths['profile_graph_topk_csv'] = os.path.dirname(args.paths['graph_csv']) + '/profile_graph_topk.csv'
        args.paths['readable_profile_graph_csv'] = os.path.dirname(args.paths['graph_csv']) + '/readable_profile_graph.csv'
        args.paths['readable_profile_graph_topk_csv'] = os.path.dirname(args.paths['graph_csv']) + '/readable_profile_graph_topk.csv'
        learned_graph_adj = pd.read_csv(args.paths['profile_graph_csv'][:-4]+'_adj.csv')
        save_graph(convert_adj2edges_wrapper(learned_graph_adj,len(args.features),dim=0, norm_flag=False),args.features,args.paths['profile_graph_csv'] ,args.paths['readable_profile_graph_csv'])

        np.fill_diagonal(learned_graph_adj.values, 0)
        learned_graph = convert_adj2edges_wrapper(learned_graph_adj,len(args.features),dim=0, norm_flag=False)

        if args.topk > 0:
            learned_graph_topk = learned_graph.clone()
            if global_top > 0:
                learned_graph_topk = learned_graph_topk[learned_graph_topk[:, 2].argsort(descending=True)][:global_top,:]
            if local_top_direction == -1:
                learned_graph_topk = convert_adj2edges_wrapper(learned_graph_topk, len(args.features), topk=args.topk ,dim=0, norm_flag=False)
                learned_graph_topk = convert_adj2edges_wrapper(learned_graph_topk, len(args.features), topk=args.topk, dim=1, norm_flag=False)
            else:
                learned_graph_topk = convert_adj2edges_wrapper(learned_graph_topk, len(args.features), topk=args.topk ,dim=local_top_direction, norm_flag=False)
            df = pd.DataFrame(learned_graph_topk,columns=['source','destination','value'])
            df['source'] = df['source'].apply(lambda x: int(x))
            df['destination'] = df['destination'].apply(lambda x: int(x))
            graph_topk = NetxGraph(df, args.features)
            save_graph(df, args.features,  args.paths['profile_graph_topk_csv'], args.paths['readable_profile_graph_topk_csv'])


            if detail_flag:
                plot_graph(args, graph_topk.graph, args.features, save_path=savepath, pos=pos, hide_nodes_flag=hide_nodes_flag, hide_isolated_flag=True,with_labels=True,colors=colors,legend_labels_text=None, markers=nodeshapes, legend_markers_text= None, communities_by=communities)
            else:
                plot_graph(args, graph_topk.graph, args.features, save_path=savepath, pos=pos, hide_nodes_flag=hide_nodes_flag,colors=colors,legend_labels_text=legend_labels_text, markers=nodeshapes, legend_markers_text= legend_markers_text, communities_by=communities)



    elif args.model =="gdn":
        args.paths["graph_topk_csv"] = os.path.dirname(args.paths['graph_csv']) + '/graph_topk.csv'
        args.paths['readable_graph_topk_csv'] = os.path.dirname(args.paths['graph_csv']) + '/readable_graph_topk.csv'
        gnn_graph = pd.read_csv(args.paths['graph_csv'])

        learned_graph_adj = convert_edges2adj(gnn_graph.values, len(args.features))
        np.fill_diagonal(learned_graph_adj, 0)
        learned_graph = convert_adj2edges_wrapper(learned_graph_adj, len(args.features), dim=0, norm_flag=False)

        if args.topk > 0:
            learned_graph_topk = learned_graph
            if global_top > 0:
                learned_graph_topk = learned_graph_topk[learned_graph_topk[:, 2].argsort(descending=True)][:global_top,:]
            if local_top_direction == -1:
                learned_graph_topk = convert_adj2edges_wrapper(learned_graph_topk, len(args.features), topk=args.topk ,dim=0, norm_flag=False)
                learned_graph_topk = convert_adj2edges_wrapper(learned_graph_topk, len(args.features), topk=args.topk, dim=1, norm_flag=False)
            else:
                learned_graph_topk = convert_adj2edges_wrapper(learned_graph_topk, len(args.features), topk=args.topk ,dim=local_top_direction, norm_flag=False)

            df = pd.DataFrame(learned_graph_topk,columns=['source','destination','value'])
            df['source'] = df['source'].apply(lambda x: int(x))
            df['destination'] = df['destination'].apply(lambda x: int(x))
            graph_topk = NetxGraph(df, args.features)
            save_graph(df, args.features,  args.paths['graph_topk_csv'], args.paths['readable_graph_topk_csv'])
            if detail_flag:
                plot_graph(args, graph_topk.graph, args.features, save_path=savepath, pos=pos, hide_nodes_flag=hide_nodes_flag)
            else:
                plot_graph(args, graph_topk.graph, args.features, save_path=savepath, pos=pos, hide_nodes_flag=hide_nodes_flag,colors=colors,legend_labels_text=legend_labels_text, markers=nodeshapes, legend_markers_text= legend_markers_text, communities_by=None)



if __name__=='__main__':
    args = get_args()
    args.no_store =True
    args.slide_stride = 1

    draw(
        args=args,
        dataset='jlab',
        model='gdn',
        graph_path='results/jlab_gdn_03-13--16-01-00/graph.csv',
        pos='fdp',
        local_top=2,
        global_top=100,
        local_top_direction=-1,
        detail_flag=False,
        savepath="./fig5_a.png"
    )

    draw(
        args=args,
        dataset='jlab',
        model='fm',
        graph_path='results/jlab_fm_03-13--13-35-29/graph_datasize/12hour/profile_graph_adj.csv',
        pos='neato',
        local_top=2,
        global_top=100,
        local_top_direction=-1,
        detail_flag=False,
        savepath="./fig5_b.png"
    )

    draw(
        args=args,
        dataset='jlab',
        model='fm',
        graph_path='results/jlab_fm_03-13--13-35-29/graph_datasize/12hour/profile_graph_adj.csv',
        pos='neato',
        local_top=2,
        global_top=100,
        local_top_direction=-1,
        detail_flag=True,
        savepath="./fig5_c.png"
    )

    draw(
        args=args,
        dataset='olcfcutsec',
        model='gdn',
        graph_path='results/olcfcutsec_gdn_05-30--01-49-01/graph.csv',
        pos='neato',
        local_top=6,
        global_top=100,
        local_top_direction=1,
        detail_flag=False,
        savepath="./fig6_a.png"
    )
    draw(
        args=args,
        dataset='olcfcutsec',
        model='fm',
        graph_path='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/12hour/profile_graph_adj.csv',
        pos='dot',
        local_top=6,
        global_top=100,
        local_top_direction=1,
        detail_flag=False,
        savepath="./fig6_b.png"
    )
    draw(
        args=args,
        dataset='olcfcutsec',
        model='fm',
        graph_path='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/12hour/profile_graph_adj.csv',
        pos='dot',
        local_top=6,
        global_top=100,
        local_top_direction=1,
        detail_flag=True,
        savepath="./fig6_c.png"
    )
