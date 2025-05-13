import torch
import pandas as pd
import numpy as np
from util.data_loader import set_miniloader
from util.net_prep import convert_adj2edges_wrapper
from datetime import timedelta
from util.net_prep import convert_adj2edges, convert_edges2adj
import os 

def profile_graph(model, args, dataloader,y_diff_instance = None):
    if not y_diff_instance.index.is_monotonic_increasing:
        raise ValueError('Index for y_diff_instance is not monotonic increasing')
    y_diff_instance = torch.from_numpy(y_diff_instance.values).to(args.device)
    graph_mean = []
    graph_std = []
    model.eval()
    with torch.no_grad():
        for i in range(len(args.features)):
            predict_delta_list = []
            global_max = []
            global_min = []
            for x, y, labels, edge_index, timestamp, instance_id, status_mask, _ ,_ in dataloader:
                x, y = [item.float().to(args.device) for item in [x, y]]
                instance_id = instance_id.long().to(args.device)
                predictions = model(x)
                disturbance = torch.zeros_like(x)
                disturbance[:, i, :] = (args.trg_epsilon*(y_diff_instance[instance_id,i])).unsqueeze(1).expand(-1, 10)
                x_with_disturbance = x + disturbance
                predictions_with_disturbance = model(x_with_disturbance)
                predictions_delta = (predictions_with_disturbance - predictions)
                predictions_delta = predictions_delta.permute(0, 2, 1)
                predictions_delta = predictions_delta.sum(-1)
                dividor = y_diff_instance[instance_id]
                mask = dividor == 0
                predictions_delta[mask] = 0
                dividor[mask] = 1e-5
                predictions_delta = torch.square(predictions_delta/dividor) 
                predict_delta_list.append(predictions_delta.cpu().numpy()) 
            predict_delta_list = np.concatenate(predict_delta_list)
            relation_mean = predict_delta_list.mean(axis=0)
            relation_std = predict_delta_list.std(axis=0)
            graph_mean.append(relation_mean)
            graph_std.append(relation_std)
    graph_mean = np.array(graph_mean)
    graph_std = np.array(graph_std)
    return graph_mean, graph_std

def profile_graph_4df(model, args, df,data, y_diff_instance):
    df = df.copy()
    df = df.reset_index(drop=True)
    dataloader = set_miniloader(df,args)
    graph_mean, graph_std = profile_graph(model, args, dataloader,y_diff_instance = y_diff_instance)
    return graph_mean, graph_std

def g2df(g):
    g = pd.DataFrame(g,columns=['source','destination','value'])
    g['source'] = g['source'].apply(lambda x: int(x))
    g['destination'] = g['destination'].apply(lambda x: int(x))
    return g

def in_degree_sequence(A):
    return np.sum(A, axis=0)

def out_degree_sequence(A):
    return np.sum(A, axis=1)


def get_similarity(g1,g2,distance='common'):
    num_nodes = g1['source'].nunique()
    if distance=='common':
        g1 = g1.sort_values('value',ascending=False)
        g2 = g2.sort_values('value',ascending=False)
        g1 = g1[:100]
        g2 = g2[:100]
        edges_1 = set(zip(g1['source'], g1['destination']))
        edges_2 = set(zip(g2['source'], g2['destination']))
        common_edges = edges_1 & edges_2
        return len(common_edges)/100
    elif distance == 'euclidean':
        v1=g1['value'].values
        v2=g2['value'].values
        return np.linalg.norm(v1-v2)
    elif distance == 'cosine':
        v1=g1['value'].values
        v2=g2['value'].values
        return np.dot(v1, v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))
    elif distance == 'degree_seq':
        adj1= convert_edges2adj(g1.to_numpy(),num_nodes)
        adj2= convert_edges2adj(g2.to_numpy(),num_nodes)
        return np.linalg.norm(in_degree_sequence(adj1) - in_degree_sequence(adj2))+ np.linalg.norm(out_degree_sequence(adj1) - out_degree_sequence(adj2))
    elif distance == 'spectral':
        adj1= convert_edges2adj(g1.to_numpy(),num_nodes)
        adj2= convert_edges2adj(g2.to_numpy(),num_nodes)
        eigenvalues_1 = np.linalg.eigvals(adj1)
        eigenvalues_2 = np.linalg.eigvals(adj2)
        return np.linalg.norm(eigenvalues_1-eigenvalues_2)
    elif distance == 'common_euclidean':
        g1_sort = g1.sort_values('value',ascending=False)
        g2_sort = g2.sort_values('value',ascending=False)
        g1_top = g1_sort[:50]
        g2_top = g2_sort[:50]
        edges_1 = set(zip(g1_top['source'], g1_top['destination']))
        edges_2 = set(zip(g2_top['source'], g2_top['destination']))
        common_edges = edges_1 & edges_2
        euclidean_val = 0
        for edge in common_edges:
            euclidean_val += (g1.loc[(g1['source']==edge[0])&(g1['destination']==edge[1]),'value'].values - g2.loc[(g2['source']==edge[0])&(g2['destination']==edge[1]),'value'].values)**2
        return euclidean_val[0]
    else:
        raise ValueError('distance should be common, euclidean or cosine')

def get_similarity_std(g1,g1_std,g2,g2_std,distance='common'):
    num_nodes = g1['source'].nunique()
    if distance=='common':
        g1 = g1.sort_values('value',ascending=False)
        g2 = g2.sort_values('value',ascending=False)
        g1 = g1[:100]
        g2 = g2[:100]
        edges_1 = set(zip(g1['source'], g1['destination']))
        edges_2 = set(zip(g2['source'], g2['destination']))
        common_edges = edges_1 & edges_2
        source = [edge[0] for edge in common_edges]
        destination = [edge[1] for edge in common_edges]
        g1_std_common = g1_std.loc[(g1_std['source'].isin(source))&(g1_std['destination'].isin(destination))]['value'].values
        g2_std_common = g2_std.loc[(g2_std['source'].isin(source))&(g2_std['destination'].isin(destination))]['value'].values
        return np.linalg.norm(g1_std_common-g2_std_common)
    elif distance == 'euclidean':
        v1=g1_std['value'].values
        v2=g2_std['value'].values
        return np.linalg.norm(v1-v2)
    elif distance == 'cosine':
        v1=g1_std['value'].values
        v2=g2_std['value'].values
        return np.dot(v1, v2) / (np.linalg.norm(v1)* np.linalg.norm(v2))
    elif distance == 'degree_seq':
        adj1= convert_edges2adj(g1_std.to_numpy(),num_nodes)
        adj2= convert_edges2adj(g2_std.to_numpy(),num_nodes)
        return np.linalg.norm(in_degree_sequence(adj1) - in_degree_sequence(adj2))+ np.linalg.norm(out_degree_sequence(adj1) - out_degree_sequence(adj2))
    elif distance == 'spectral':
        adj1= convert_edges2adj(g1_std.to_numpy(),num_nodes)
        adj2= convert_edges2adj(g2_std.to_numpy(),num_nodes)
        eigenvalues_1 = np.linalg.eigvals(adj1)
        eigenvalues_2 = np.linalg.eigvals(adj2)
        return np.linalg.norm(eigenvalues_1-eigenvalues_2)
    elif distance == 'common_euclidean':
        g1_sort = g1_std.sort_values('value',ascending=False)
        g2_sort = g2_std.sort_values('value',ascending=False)
        g1_top = g1_sort[:50]
        g2_top = g2_sort[:50]
        edges_1 = set(zip(g1_top['source'], g1_top['destination']))
        edges_2 = set(zip(g2_top['source'], g2_top['destination']))
        common_edges = edges_1 & edges_2
        euclidean_val = 0
        for edge in common_edges:
            euclidean_val += (g1_std.loc[(g1_std['source']==edge[0])&(g1_std['destination']==edge[1]),'value'].values - g2_std.loc[(g2_std['source']==edge[0])&(g2_std['destination']==edge[1]),'value'].values)**2
        return euclidean_val[0]
    else:
        raise ValueError('distance should be common, euclidean or cosine')



def get_len(g,distance='l2'):
    if distance=='l2':
        v=g['value'].values
        v=v[:50]
        return np.linalg.norm(v)
    else:
        raise ValueError('distance should be l2')


class Graph_extracter():
    def __init__(self, model, args, data):
        self.model =model
        self.args = args
        self.data = data
        if hasattr(args,'tol_min'):
            self.whole = self.data.test
            self.tol_min = self.args.tol_min
        else:
            self.whole = pd.concat([self.data.train, self.data.val, self.data.test]).reset_index(drop=True)
        self.cur_graph_mean = None
        self.cur_graph_std = None
        
        self.tp_sim_list = []
        self.fp_sim_list = []
        self.tp_global_sim_list = []
        self.fp_global_sim_list = []
        
        save_subdir_first = os.path.normpath(args.save_subdir).split(os.sep)[0]
        self.saved_graph_dir = os.path.abspath(os.path.dirname(args.paths['best_pt']))+'/cache_graph/'+save_subdir_first
        if not os.path.exists(self.saved_graph_dir) and args.no_pCache==False:
            os.makedirs(self.saved_graph_dir)
        if set(data.val['instance_id'].unique()).issubset(data.train['instance_id'].unique()) and set(data.test['instance_id'].unique()).issubset(data.train['instance_id'].unique()):
            y_q75_instance = data.train.groupby('instance_id').quantile(0.75)[args.features]
            y_q25_instance = data.train.groupby('instance_id').quantile(0.25)[args.features]
        else:
            df_all = pd.concat([data.train,data.val,data.test])
            y_q75_instance = df_all.groupby('instance_id').quantile(0.75)[args.features]
            y_q25_instance = df_all.groupby('instance_id').quantile(0.25)[args.features]

        self.y_diff_instance = (y_q75_instance - y_q25_instance).sort_index()
        if args.dataset=='jlab':
            global_g_path ='results/jlab_fm_03-13--13-35-29/graph_datasize/24hour/profile_graph_adj.csv'
        elif args.dataset=='olcfcutsec':
            global_g_path ='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/24hour/profile_graph_adj.csv'
        else:
            print('[Warning] Use wrong global_g_path')
            global_g_path ='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/24hour/profile_graph_adj.csv'
        print('>>>>> Use global graph:', global_g_path)
        self.g_global = pd.read_csv(global_g_path)
        self.g_global = pd.DataFrame(convert_adj2edges(self.g_global),columns=['source','destination','value'])
        self.g_global['source'] = self.g_global['source'].apply(lambda x: int(x))
        self.g_global['destination'] = self.g_global['destination'].apply(lambda x: int(x))


        
    def profile_graph_4hours(self, df, time_win = 1, history_shift=0,instance=None):


        if instance !=None:
            idxs = df['timestamp'].idxmax()   
        elif 'instance' in df.columns:
            idxs = df.groupby('instance')['timestamp'].idxmax().values   
        else:
            raise ValueError('instance column not found in the dataframe')
        idxs = np.array(idxs)
        if instance!=None:
            timestamp = self.data.test.iloc[idxs]['timestamp']
        elif 'instance' in df.columns:
            instance_col_id = self.data.test.columns.get_loc('instance')
            timestamp_col_id = self.data.test.columns.get_loc('timestamp')
            last_row = self.data.test.iloc[idxs,[timestamp_col_id,instance_col_id]].set_index('instance')['timestamp'].reset_index()
            instance = last_row['instance'].iloc[0]
            timestamp = last_row['timestamp'].iloc[0]
        else:
            raise ValueError('instance column not found in the dataframe')
        
        mask = (self.whole['instance']==instance)&(self.whole['timestamp']<=(timestamp- timedelta(hours=history_shift)))&(self.whole['timestamp']>= (timestamp- timedelta(hours=time_win)- timedelta(hours=history_shift)))
        saved_graph_path = self.saved_graph_dir+'/{}_{}_{}'.format(instance,(timestamp- timedelta(hours=time_win)- timedelta(hours=history_shift)).strftime('%Y-%m-%d_%H-%M-%S'),(timestamp- timedelta(hours=history_shift)).strftime('%Y-%m-%d_%H-%M-%S'))
        if self.whole.loc[mask].shape[0] <=self.args.slide_win:
            mask = (self.whole['instance']==instance)&(self.whole['timestamp']<=(timestamp- timedelta(hours=time_win)- timedelta(hours=history_shift)))&(self.whole['timestamp']>= (timestamp- timedelta(hours=time_win)- timedelta(hours=time_win)- timedelta(hours=history_shift)))
            saved_graph_path = self.saved_graph_dir+'/{}_{}_{}'.format(instance, (timestamp- timedelta(hours=time_win)- timedelta(hours=time_win)- timedelta(hours=history_shift)).strftime('%Y-%m-%d_%H-%M-%S'), (timestamp- timedelta(hours=time_win)- timedelta(hours=history_shift)).strftime('%Y-%m-%d_%H-%M-%S'))
        if self.whole.loc[mask].shape[0] <=self.args.slide_win:
            return np.array([-1]), np.array([-1])
        if os.path.exists(saved_graph_path):
            print('>>>>> Use TRG:', saved_graph_path)
            self.cur_graph_mean = np.load(saved_graph_path+'/graph_mean.npy')
            self.cur_graph_std = np.load(saved_graph_path+'/graph_std.npy')
        else:
            print('>>>>> Use TRG:', saved_graph_path)
            self.cur_graph_mean, self.cur_graph_std = profile_graph_4df(self.model, self.args, self.whole.loc[mask] , self.data, y_diff_instance = self.y_diff_instance)
            if self.args.no_pCache==False:
                if not os.path.exists(saved_graph_path):
                    os.makedirs(saved_graph_path)
                np.save(saved_graph_path+'/graph_mean',self.cur_graph_mean)
                np.save(saved_graph_path+'/graph_std',self.cur_graph_std)

        return self.cur_graph_mean, self.cur_graph_std

    def convert_adj2edges_topk(self, g_adj, nodes_num=None,topk=-1, dim=0, norm_flag=False):
        if nodes_num==None: nodes_num=len(self.args.features)
        return convert_adj2edges_wrapper(g_adj,nodes_num=nodes_num,topk=topk,dim=dim, norm_flag=norm_flag)
    
    def id2feature(self, id):
        return self.args.features[id]

    def get_global_similarity(self, g):
        return get_similarity(g,self.g_global)