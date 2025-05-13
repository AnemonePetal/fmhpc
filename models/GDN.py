import numpy as np
import torch
import torch.nn as nn
import time
from util.time import *
from util.env import *
from util.net_prep import convert_adj2edges_wrapper
import torch.nn.functional as F
import pandas as pd
from .graph_layer import GraphLayer
from torch.utils.tensorboard import SummaryWriter
from util.postprocess import align_df
from util.save import save_result,save_graph
from util.Data import filter_df
from util.argparser import save_args
from util.data_loader import set_dataloader
import optuna
from util.deterministic import set_deterministic
from util import regression_score
from util.anomaly_dtask import anomaly_score
from util import regression_score

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out

class EmbeddingLayer(nn.Module):
    def __init__(self, node_num, dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(node_num, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, x):
        out = self.embedding(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class CurEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, dim):
        super(CurEmbeddingLayer, self).__init__()
        self.fc = nn.Linear(input_dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = out.permute(0,2,1)
        out = self.bn(out)
        out = out.permute(0,2,1)
        out = self.relu(out)
        return out

class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out = self.gnn(x, edge_index, embedding, return_attention_weights=False)
  
        out = self.bn(out)
        
        return out

class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20, train_config=None):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        self.init_weighted_adj =None
        self.embed_dim = dim
        self.embedding = nn.Embedding(node_num, self.embed_dim)
        self.cur_embedding = nn.Linear(input_dim, self.embed_dim)

        self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+self.embed_dim, heads=1) for i in range(edge_set_num)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)
        self.train_config = train_config
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_uniform_(self.embedding.weight)


    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device
        if len(x.shape) == 2:
            batch_num, node_num = x.shape
            all_feature = 1
        elif len(x.shape) == 3:
            batch_num, node_num, all_feature = x.shape
        else:
            raise Exception("x shape error: it should be 3")


        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
                        
            all_embeddings = self.embedding(torch.arange(node_num).to(device))            

            weights_arr = all_embeddings.detach().clone()


            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / torch.clamp(normed_mat, min=1e-10)




            processed_cos_ji_mat = cos_ji_mat
            topk_num = self.topk
            topk_values_ji, topk_indices_ji = torch.topk(processed_cos_ji_mat, topk_num, dim=-1) 
            

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            gated_edge_value = topk_values_ji.flatten().unsqueeze(0)
            gated_edge = torch.cat((gated_edge_index, gated_edge_value), dim=0)
            gated_edge = gated_edge[:, gated_edge[2,:] != 0]
            gated_edge_index = gated_edge[:2, :].int()



            all_embeddings = all_embeddings.repeat(batch_num, 1)
            x_flat = x.view(-1, all_feature).contiguous()
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x_flat, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out)

            self.learned_graph = cos_ji_mat

        gcn_outs = torch.cat(gcn_outs, dim=1)
        gcn_outs = gcn_outs.view(batch_num, node_num, -1)


        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(gcn_outs, self.embedding(indexes))

        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)
        
        out = self.dp(out)
        predictions = self.out_layer(out)
        predictions = predictions.view(-1, node_num)

        return predictions, out

    def generate_representation(self, data: torch.Tensor, entryid: torch.Tensor) -> torch.Tensor:
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device
        if len(x.shape) == 2:
            batch_num, node_num = x.shape
            all_feature = 1
        elif len(x.shape) == 3:
            batch_num, node_num, all_feature = x.shape
        else:
            raise Exception("x shape error: it should be 3")


        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))            
            weights_arr = all_embeddings.detach().clone()
            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / torch.clamp(normed_mat, min=1e-10)

            processed_cos_ji_mat = cos_ji_mat
            topk_num = self.topk
            topk_values_ji, topk_indices_ji = torch.topk(processed_cos_ji_mat, topk_num, dim=-1) 
            
            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            gated_edge_value = topk_values_ji.flatten().unsqueeze(0)
            gated_edge = torch.cat((gated_edge_index, gated_edge_value), dim=0)
            gated_edge = gated_edge[:, gated_edge[2,:] != 0]
            gated_edge_index = gated_edge[:2, :].int()

            all_embeddings = all_embeddings.repeat(batch_num, 1)
            x_flat = x.view(-1, all_feature).contiguous()
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x_flat, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            gcn_outs.append(gcn_out)

            self.learned_graph = cos_ji_mat

        gcn_outs = torch.cat(gcn_outs, dim=1)
        gcn_outs = gcn_outs.view(batch_num, node_num, -1)
        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(gcn_outs, self.embedding(indexes))
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)
        out = self.dp(out)
        
        for index, mod in enumerate(self.out_layer.mlp):
            if index == len(self.out_layer.mlp)-1:
                break
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)
        out = out.permute(0,2,1)
        out = out.view(batch_num, -1)
        return out

        
class GDN_wrapper(object,):
    def __init__(self, args,data,tensorboard=False):
        if tensorboard and args.store and args.load_model_path=='':
            self.writer = SummaryWriter(os.path.dirname(args.paths['test_re'])+'/tensorboard')
        if args.deterministic:
            set_deterministic(args)
        set_dataloader(data,args)
        self.args= args
        self.data = data
        self._setup_model()
    
    def _setup_model(self):
        self.model = GDN(edge_index_sets = self.data.edge_index_sets, 
                        node_num = len(self.args.features),
                        dim = self.args.dim,
                        input_dim= self.args.slide_win,
                        out_layer_num=self.args.out_layer_num,
                        out_layer_inter_dim=self.args.out_layer_inter_dim,
                        topk= len(self.args.features),
                        train_config=self.args).to(self.args.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    def predict(self,dataloader):
        self.model.eval()
        with torch.no_grad():
            loss_func = nn.MSELoss(reduction='mean')
            device = get_device()

            test_loss = []
            test_predicted_list = []
            for x, y, labels, edge_index, timestamp,instance_id, status_mask, _ ,_ in dataloader:
                x, y, edge_index = [item.to(device).float() for item in [x, y, edge_index]]
                if y.shape[1] != 1:
                    raise Exception('Prediction Time window should be 1')
                y = y.squeeze(1)
                predictions,_ = self.model(x, edge_index)
                loss = loss_func(predictions, y).cpu()       
                test_loss.append(loss.item())
                test_predicted_list.append(predictions.cpu().numpy())
            test_predicted_list = np.concatenate(test_predicted_list)
            test_loss = np.mean(test_loss)
        return test_loss, test_predicted_list

    def train(self):
        print("Training GDN...")
        train_loss_list = []
        epoch_times = []
        min_loss = float('inf')
        best_model_state = None
        stop_improve_count = 0
        for step in range(self.args.epoch):
            self.model.train()
            train_loss = []
            start_time_epoch = time.time()

            for  x, y, labels, edge_index, timestamp,instance_id, status_mask,_ , _ in self.data.train_dataloader:
                x, y, edge_index = [item.float().to(self.args.device) for item in [x, y, edge_index]]
                if y.shape[1] != 1:
                    raise Exception('Prediction Time window should be 1')
                y = y.squeeze(1)                
                self.optimizer.zero_grad()
                predictions,_ = self.model(x, edge_index)
                loss = F.mse_loss(predictions, y, reduction='mean')
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()                
                loss.detach()
                train_loss.append(loss.item())
                del loss
                del predictions
            mean_loss = np.mean(train_loss)
            train_loss_list.append(mean_loss)
            epoch_time = time.time() - start_time_epoch
            epoch_times.append(epoch_time)
            print('Epoch: {}, loss: {}, lapsed time: {}'.format(step, mean_loss, epoch_time))
            if hasattr(self, 'writer'):
                self.writer.add_scalar('total_loss', mean_loss, step)
            if self.data.val_dataloader is not None:
                val_loss, val_result = self.predict(self.data.val_dataloader)
                if hasattr(self, 'optuna_trial'):
                    self.optuna_trial.report(val_loss, step)
                if self.args.optuna_prune:
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()
                if val_loss < min_loss:
                    torch.save(self.model.state_dict(), self.args.paths['cache_pt'])
                    min_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1
                if stop_improve_count >= self.args.early_stop_win:
                    break
        print('Whole lapsed time: {}'.format(np.sum(epoch_times)))
        self.model.load_state_dict(torch.load(self.args.paths['cache_pt']))
        if self.args.store:
            if os.path.exists(self.args.paths['cache_pt']):
                os.rename(self.args.paths['cache_pt'], self.args.paths['best_pt'])
            loss_df = pd.DataFrame(train_loss_list, columns=['loss'])
            loss_df.to_csv(self.args.paths['loss_csv'], index=False)

        return train_loss_list, best_model_state

    def score(self):
        if not os.path.exists(self.args.paths['train_re']):
            _, self.data.train_re = self.predict(self.data.train_dataloader)
            self.data.train_re = save_result(self.args, self.data, result_type='train',save2disk=self.args.pCache or self.args.attack)
        else:
            self.data.train_re = pd.read_csv(self.args.paths['train_re'])
            self.data.train_re['timestamp'] = pd.to_datetime(self.data.train_re['timestamp'])
            self.data.train_re = filter_df(self.data.train_re, self.args.data_filter)
        if not os.path.exists(self.args.paths['test_re']):
            _, self.data.test_re = self.predict(self.data.test_dataloader)
            self.data.test_re = save_result(self.args, self.data, result_type='test',save2disk=self.args.pCache)
        else:
            self.data.test_re = pd.read_csv(self.args.paths['test_re'])
            self.data.test_re['timestamp'] = pd.to_datetime(self.data.test_re['timestamp'])
            self.data.test_re = filter_df(self.data.test_re, self.args.data_filter)

        if not os.path.exists(self.args.paths['val_re']):
            _, self.data.val_re = self.predict(self.data.val_dataloader)
            self.data.val_re = save_result(self.args, self.data, result_type='val',save2disk=self.args.pCache or self.args.attack)
        else:
            self.data.val_re = pd.read_csv(self.args.paths['val_re'])
            self.data.val_re['timestamp'] = pd.to_datetime(self.data.val_re['timestamp'])
            self.data.val_re = filter_df(self.data.val_re, self.args.data_filter)
        
        self.data.train = align_df(self.data.train_re, self.data.train, self.data.train_dataset.rang,self.args)
        self.data.test = align_df(self.data.test_re, self.data.test, self.data.test_dataset.rang,self.args)
        self.data.val = align_df(self.data.val_re, self.data.val, self.data.val_dataset.rang,self.args)
        
        if not (hasattr(self.args, 'load_model_path') and  self.args.load_model_path != '' and self.args.reload_args):
            save_args(self.args)

        self.args.paths["graph_topk_csv"] = os.path.dirname(self.args.paths['graph_csv']) + '/graph_topk.csv'
        self.args.paths['readable_graph_topk_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/readable_graph_topk.csv'
        if self.args.store:
            if hasattr(self.model, 'learned_graph') and self.model.learned_graph != None and not os.path.exists(self.args.paths['graph_topk_csv']):
                learned_graph = convert_adj2edges_wrapper(self.model.learned_graph,len(self.args.features),dim=0, norm_flag=False)
                save_graph(learned_graph,self.args.features, self.args.paths['graph_csv'] ,self.args.paths['readable_graph_csv'])
                learned_graph_topk = learned_graph
                learned_graph_topk_df = pd.DataFrame(learned_graph_topk,columns=['source','destination','value'])
                learned_graph_topk_df['source'] = learned_graph_topk_df['source'].apply(lambda x: int(x))
                learned_graph_topk_df['destination'] = learned_graph_topk_df['destination'].apply(lambda x: int(x))
                save_graph(learned_graph_topk_df, self.args.features,  self.args.paths['graph_topk_csv'], self.args.paths['readable_graph_topk_csv']) 

        regression_score.get_score(self.data, self.args, draw_eval=True,save2disk=self.args.store,data_filter=self.args.data_filter)
        if self.args.dtask == 'anomaly':
            anomaly_score.get_score(self.data, self.args, graph_path=self.args.paths['graph_topk_csv'] ,draw_eval=True,save2disk=self.args.store, pCache=self.args.pCache,data_filter=self.args.data_filter)

    def load(self):
        if not hasattr(self.args, 'load_model_path') or self.args.load_model_path == '':
            return True
        else:
            self.model.load_state_dict(torch.load(self.args.paths['best_pt']))