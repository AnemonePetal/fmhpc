import numpy as np
import torch
import torch.nn as nn
import time
from util.time import *
from util.env import *
from util.net_prep import convert_adj2edges_wrapper
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import copy
from util.postprocess import align_df
from util.save import save_result,save_graph
from util.Data import filter_df
from util.argparser import save_args
from util.plot import plot_graph_wrapper
from util.data_loader import set_dataloader
import optuna
from util.deterministic import set_deterministic
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
        self.layer_num = layer_num
        super(OutLayer, self).__init__()
        if layer_num >1:
            self.bn = nn.BatchNorm1d(in_num)
        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear( in_num , 1))
            else:
                layer_in_num = in_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.ReLU())
                modules.append(nn.Linear( inter_num,layer_in_num))

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for index, mod in enumerate(self.mlp):
            if index == len(self.mlp)-1 and self.layer_num != 1:
                out = (out+x).permute(0,2,1)
                out = self.bn(out)
                out = out.permute(0,2,1)
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out

class FMTMLayer(nn.Module):
    def __init__(self, edge_index_sets, node_num, hnode_num=332, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, train_config=None):

        super(FMTMLayer, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]

        self.init_weighted_adj =None
        embed_dim = dim

        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        self.lin = nn.Linear(input_dim, embed_dim, bias=True)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()

        self.node_embedding = None
        self.learned_graph = None
        self.out_layer = OutLayer(embed_dim , node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)
        self.train_config = train_config


    def forward(self, data):
        x = data.clone().detach()


        device = data.device
        if len(x.shape) == 2:
            batch_num, node_num = x.shape
            all_feature = 1
        elif len(x.shape) == 3:
            batch_num, node_num, all_feature = x.shape
        else:
            raise Exception("x shape error: it should be 3")


        out = self.lin(x)
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        predictions = self.out_layer(out)


        predictions = predictions.view(-1, node_num)

        return predictions, out

    def generate_representation(self, data: torch.Tensor, entryid: torch.Tensor) -> torch.Tensor:
        x = data.clone().detach()
        device = data.device
        if len(x.shape) == 2:
            batch_num, node_num = x.shape
            all_feature = 1
        elif len(x.shape) == 3:
            batch_num, node_num, all_feature = x.shape
        else:
            raise Exception("x shape error: it should be 3")
        out = self.lin(x)
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)
        x = self.dp(out)
        out = x
        for index, mod in enumerate(self.out_layer.mlp):
            if index == len(self.out_layer.mlp)-1 and self.out_layer.layer_num != 1:
                out = (out+x).permute(0,2,1)
                out = self.out_layer.bn(out)
                out = out.permute(0,2,1)
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

class FM_TM_wrapper(object,):
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
        self.model = FMTMLayer(edge_index_sets = self.data.edge_index_sets, 
                        node_num = len(self.args.features),
                        hnode_num= len(self.data.train['instance'].unique()),
                        dim = self.args.dim,
                        input_dim= self.args.slide_win,
                        out_layer_num=self.args.out_layer_num,
                        out_layer_inter_dim=self.args.out_layer_inter_dim,
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
            for x, y, labels, edge_index, timestamp, instance_id, status_mask, _ ,_ in dataloader:
                x, y = [item.float().to(self.args.device) for item in [x, y]]
                instance_id = instance_id.int().to(self.args.device)

                predictions,_ = self.model(x)
                loss = loss_func(predictions, y).cpu()       
                test_loss.append(loss.item())
                test_predicted_list.append(predictions.cpu().numpy())
            test_predicted_list = np.concatenate(test_predicted_list)
            test_loss = np.mean(test_loss)
        return test_loss, test_predicted_list

    def train(self):
        print("Training FM_TM...")
        train_loss_list = []
        epoch_times = []
        min_loss = float('inf')
        best_model_state = None
        stop_improve_count = 0
        for step in range(self.args.epoch):
            self.model.train()
            train_loss = []
            start_time_epoch = time.time()
            for  x, y, labels, edge_index, timestamp, instance_id, status_mask,_ , _ in self.data.train_dataloader:
                x, y = [item.float().to(self.args.device) for item in [x, y]]
                instance_id = instance_id.int().to(self.args.device)
                self.optimizer.zero_grad()
                predictions,_ = self.model(x)
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

    def profile_graph(self, model, args, dataloader):
        graph_mean = []
        graph_std = []
        self.model.eval()
        with torch.no_grad():
            for i in range(len(args.features)):
                test_predicted_list = []
                for x, y, labels, edge_index, timestamp, instance_id, status_mask, _ ,_ in dataloader:
                    x, y = [item.float().to(self.args.device) for item in [x, y]]
                    instance_id = instance_id.int().to(self.args.device)
                    predictions,_ = self.model(x)

                    disturbance = torch.zeros_like(x)
                    disturbance[:, i, :] = 5e-2
                    x_with_disturbance = x + disturbance
                    predictions_with_disturbance,_ = self.model(x_with_disturbance)

                    predictions_delta = predictions_with_disturbance - predictions
                    test_predicted_list.append(predictions_delta.cpu().numpy())
                test_predicted_list = np.concatenate(test_predicted_list)
                relation_mean = test_predicted_list.mean(axis=0)
                graph_mean.append(relation_mean)
        graph_mean = np.array(graph_mean).T
        graph_mean = graph_mean / abs(graph_mean[graph_mean!=0]).min()
        return graph_mean

    def score(self):
        if not os.path.exists(self.args.paths['train_re']):
            _, self.data.train_re = self.predict(self.data.train_dataloader)
            self.data.train_re = save_result(self.args, self.data, result_type='train',save2disk= self.args.pCache or self.args.attack)
        else:
            self.data.train_re = pd.read_csv(self.args.paths['train_re'])
            self.data.train_re['timestamp'] = pd.to_datetime(self.data.train_re['timestamp'])
            self.data.train_re = filter_df(self.data.train_re, self.args.data_filter)
        if not os.path.exists(self.args.paths['test_re']):
            _, self.data.test_re = self.predict(self.data.test_dataloader)
            self.data.test_re = save_result(self.args, self.data, result_type='test',save2disk= self.args.pCache)
        else:
            self.data.test_re = pd.read_csv(self.args.paths['test_re'])
            self.data.test_re['timestamp'] = pd.to_datetime(self.data.test_re['timestamp'])
            self.data.test_re = filter_df(self.data.test_re, self.args.data_filter)

        if not os.path.exists(self.args.paths['val_re']):
            _, self.data.val_re = self.predict(self.data.val_dataloader)
            self.data.val_re = save_result(self.args, self.data, result_type='val',save2disk= self.args.pCache or self.args.attack)
        else:
            self.data.val_re = pd.read_csv(self.args.paths['val_re'])
            self.data.val_re['timestamp'] = pd.to_datetime(self.data.val_re['timestamp'])
            self.data.val_re = filter_df(self.data.val_re, self.args.data_filter)
        
        self.data.train = align_df(self.data.train_re, self.data.train, self.data.train_dataset.rang,self.args)
        self.data.test = align_df(self.data.test_re, self.data.test, self.data.test_dataset.rang,self.args)
        self.data.val = align_df(self.data.val_re, self.data.val, self.data.val_dataset.rang,self.args)
        

        if not (hasattr(self.args, 'load_model_path') and  self.args.load_model_path != '' and self.args.reload_args):
            save_args(self.args)
        if self.args.store:
            self.args.paths['profile_graph_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/profile_graph.csv'
            self.args.paths['readable_profile_graph_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/readable_profile_graph.csv'
            if True:
                self.model.profile_graph_adj = self.profile_graph(self.model, self.args, self.data.val_dataloader)
                pd.DataFrame(self.model.profile_graph_adj).to_csv(self.args.paths['profile_graph_csv'][:-4]+'_adj.csv',index=False)
                learned_graph = convert_adj2edges_wrapper(self.model.profile_graph_adj,len(self.args.features),dim=0, norm_flag=False)
                save_graph(learned_graph,self.args.features, self.args.paths['profile_graph_csv'] ,self.args.paths['readable_profile_graph_csv'])
            if hasattr(self.args.paths, 'profile_graph_csv'):
                plot_graph_wrapper(self.args.paths['profile_graph_csv'],self.args.features)



        regression_score.get_score(self.data, self.args, draw_eval=True,save2disk=self.args.store,data_filter=self.args.data_filter)

    def load(self):
        if not hasattr(self.args, 'load_model_path') or self.args.load_model_path == '':
            return True
        else:
            self.model.load_state_dict(torch.load(self.args.paths['best_pt']))