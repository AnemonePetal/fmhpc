import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from util.postprocess import align_df
from util.save import save_result,save_graph
from util.Data import filter_df,slice_df_by_time_range
from util.argparser import save_args
from util.data_loader import set_dataloader
import optuna
from util.env import *
from util.deterministic import set_deterministic
from util.net_prep import convert_adj2edges_wrapper
from util.anomaly_dtask import anomaly_score
from util import regression_score
import time
from util.extract_graph import profile_graph,profile_graph_4df,Graph_extracter
from util.time_range import str2time,shift_time_str,shift_time_range
from pathlib import Path
from util.extract_graph import get_similarity
from util.net_prep import convert_adj2edges
from torch.profiler import profile, record_function, ProfilerActivity

class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        output = self.bn(x)
        output = output.squeeze(1)
        return output

class TimeMixerLayer(nn.Module):
    def __init__(self, width_time: int, dropout: float):
        super(TimeMixerLayer, self).__init__()
        self.norm = NormLayer()
        self.lin = nn.Linear(in_features=width_time, out_features=width_time)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = torch.transpose(y, 1, 2)
        y = self.lin(y)
        y = self.act(y)
        y = torch.transpose(y, 1, 2)
        y = self.dropout(y)
        return x + y

class FeatMixerLayer(nn.Module):
    def __init__(self, width_feats: int, width_feats_hidden: int, dropout: float):
        super(FeatMixerLayer, self).__init__()
        self.norm = NormLayer()
        self.lin_1 = nn.Linear(in_features=width_feats, out_features=width_feats_hidden)
        self.lin_2 = nn.Linear(in_features=width_feats_hidden, out_features=width_feats)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.lin_1(y)
        y = self.act(y)
        y = self.dropout_1(y)
        y = self.lin_2(y)
        y = self.dropout_2(y)
        return x + y

class MixerLayer(nn.Module):
    def __init__(self, input_length: int, no_feats: int, feat_mixing_hidden_channels: int, dropout: float):
        super(MixerLayer, self).__init__()
        self.time_mixing = TimeMixerLayer(width_time=input_length, dropout=dropout)
        self.feat_mixing = FeatMixerLayer(width_feats=no_feats, width_feats_hidden=feat_mixing_hidden_channels, dropout=dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.time_mixing(x)
        y = self.feat_mixing(y)
        return y

class OutLayer(nn.Module):
    def __init__(self, input_length: int, forecast_length: int):
        super(OutLayer, self).__init__()
        self.lin = nn.Linear(in_features=input_length, out_features=forecast_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.transpose(x, 1, 2)
        y = self.lin(y)
        y = torch.transpose(y, 1, 2)
        return y

class MultiMixerLayers(nn.Module):

    def __init__(self, input_length: int, forecast_length: int, no_feats: int, feat_mixing_hidden_channels: int, no_mixer_layers: int, dropout: float):
        super(MultiMixerLayers, self).__init__()
        self.temp_proj = OutLayer(input_length=input_length, forecast_length=forecast_length)
        mixer_layers = []
        for _ in range(no_mixer_layers):
            mixer_layers.append(MixerLayer(input_length=input_length, no_feats=no_feats, feat_mixing_hidden_channels=feat_mixing_hidden_channels, dropout=dropout))
        self.mixer_layers = nn.ModuleList(mixer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.temp_proj(x)
        return x

    def generate_representation(self, x: torch.Tensor) -> torch.Tensor:
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        return x

class Mixer(nn.Module):
    """Include Reversible instance normalization https://openreview.net/pdf?id=cGDAkQo1C0p
    """    

    def __init__(self, input_length: int, forecast_length: int, no_feats: int, feat_mixing_hidden_channels: int, no_mixer_layers: int,  dropout: float, eps: float = 1e-8):
        super(Mixer, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(no_feats))
        self.shift = nn.Parameter(torch.zeros(no_feats))

        self.ts = MultiMixerLayers(
            input_length=input_length, 
            forecast_length=forecast_length, 
            no_feats=no_feats, 
            feat_mixing_hidden_channels=feat_mixing_hidden_channels,
            no_mixer_layers=no_mixer_layers,
            dropout=dropout
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.scale + self.shift

        x = self.ts(x)

        x = (x - self.shift) / self.scale
        x = x * torch.sqrt(var + self.eps) + mean
        return x
    
    def generate_representation(self, x: torch.Tensor, entryid: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.scale + self.shift

        x = self.ts.generate_representation(x)
        x = (x - self.shift) / self.scale
        x = x * torch.sqrt(var + self.eps) + mean
        x = x.permute(0, 2, 1)
        return x

class FM_wrapper(object,):
    def __init__(self, args,data,tensorboard=False):
        if tensorboard and args.store and args.load_model_path=='':
            self.writer = SummaryWriter(os.path.dirname(args.paths['test_re'])+'/tensorboard')
        if args.deterministic:
            set_deterministic(args)
        start_time = time.time()
        set_dataloader(data,args)
        # print('Dataloader init time: {}'.format(time.time()-start_time))
        self.args= args
        self.data = data
        self._setup_model()
    
    def _setup_model(self):
        self.model = Mixer(
                        input_length=self.args.slide_win, 
                        forecast_length=self.args.pred_win, 
                        no_feats= len(self.args.features), 
                        feat_mixing_hidden_channels= self.args.out_layer_inter_dim, 
                        no_mixer_layers=self.args.out_layer_num,  
                        dropout=0.2,
                    ).to(self.args.device)
        
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

                predictions = self.model(x)
                loss = loss_func(predictions, y).cpu()       
                test_loss.append(loss.item())
                predictions_np = predictions.cpu().numpy()
                predictions_np = predictions_np.reshape(-1, predictions_np.shape[2])
                test_predicted_list.append(predictions_np)
            test_predicted_list = np.concatenate(test_predicted_list)
            test_loss = np.mean(test_loss)
        return test_loss, test_predicted_list

    def train(self):
        print("Training FM...")
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
                predictions = self.model(x)
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
                if self.writer is not None:
                    self.writer.add_scalar('val_loss', val_loss, step)
                    self.writer.flush()

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

    def train_speed_eval(self):
        print("Profile: Training FM...")
        train_loss_list = []
        epoch_times = []
        min_loss = float('inf')
        best_model_state = None
        stop_improve_count = 0

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        print('Profiler activities: {}'.format(activities))
        profiler_dir = './cache/profiler_logs'
        os.makedirs(profiler_dir, exist_ok=True)

        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for step in range(3):
                self.model.train()
                train_loss = []
                start_time_epoch = time.time()
                for  x, y, labels, edge_index, timestamp, instance_id, status_mask,_ , _ in self.data.train_dataloader:
                    with record_function("data_transfer"):
                        x, y = [item.float().to(self.args.device) for item in [x, y]]
                        instance_id = instance_id.int().to(self.args.device)
                    with record_function("model_train"):
                        self.optimizer.zero_grad()
                        predictions = self.model(x)
                        loss = F.mse_loss(predictions, y, reduction='mean')
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()                
                        loss.detach()
                        train_loss.append(loss.item())
                    del loss
                    del predictions
                    prof.step()
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

    def profile_graph_wrapper(self, model, args, dataloader,data):
        if set(data.val_gt['instance_id'].unique()).issubset(data.train_gt['instance_id'].unique()) and set(data.test_gt['instance_id'].unique()).issubset(data.train_gt['instance_id'].unique()):
            y_q75_instance = data.train_gt.groupby('instance_id').quantile(0.75)[args.features]
            y_q25_instance = data.train_gt.groupby('instance_id').quantile(0.25)[args.features]
        else:
            df_all = pd.concat([data.train_gt,data.val_gt,data.test_gt])
            y_q75_instance = df_all.groupby('instance_id').quantile(0.75)[args.features]
            y_q25_instance = df_all.groupby('instance_id').quantile(0.25)[args.features]

        y_diff_instance = (y_q75_instance - y_q25_instance).sort_index()

        graph_mean, graph_std = profile_graph(model=model, args=args, dataloader=dataloader, y_diff_instance = y_diff_instance)
        return graph_mean, graph_std

    
    def evolve_graph(self,farmnode = 'a33n13'):
        whole_data = pd.concat([self.data.train,self.data.val,self.data.test])
        whole_data = whole_data.reset_index(drop=True)
        if self.args.dataset=='jlab':
            start_time = '2023-05-21 04:00:00'
            end_time = '2023-05-23 04:00:00'
        else:
            start_time = '2020-01-01 00:00:00'
            end_time = '2020-01-03 21:00:00'
        whole_data = whole_data[whole_data['instance']==farmnode]
        whole_data = slice_df_by_time_range(whole_data,[[start_time,end_time]])
        whole_data = whole_data.reset_index(drop=True)

        y_q75_instance = self.data.train.groupby('instance_id').quantile(0.75)[self.args.features]
        y_q25_instance = self.data.train.groupby('instance_id').quantile(0.25)[self.args.features]
        y_diff_instance = (y_q75_instance - y_q25_instance).sort_index()
        graph_time_win = 1800
        cur_time_range = [[start_time,shift_time_str(start_time, pd.Timedelta(seconds=graph_time_win), direction=1)]]

        pos = 'neato'
        topk = len(self.args.features)
        global_top = -1
        global_top = 100
        graph_dir = os.path.join(os.path.dirname(self.args.paths['test_re']),'graph_evolve',farmnode+'_'+str(global_top)+'s')
        while str2time(cur_time_range[0][1]) <= str2time(end_time):
            cur_graph_dir = os.path.join(graph_dir,cur_time_range[0][0].replace(' ','_')[5:])
            cur_data = slice_df_by_time_range(whole_data,cur_time_range)
            print('data size:',cur_data.shape[0])
            empty_row= cur_data.tail(1).copy()
            cur_data = pd.concat([cur_data,empty_row])
            if cur_data.shape[0] <= self.args.slide_win:
                print('[WARN] Skip one time range: the records are less than one time window.')
                cur_time_range = shift_time_range(cur_time_range,pd.Timedelta(seconds=graph_time_win),direction=1)
                continue
            g_adj,g_std = profile_graph_4df(self.model, self.args, cur_data , self.data, y_diff_instance = y_diff_instance)
            Path(cur_graph_dir).mkdir(parents=True, exist_ok=True)
            path_profile_graph_csv = cur_graph_dir + '/profile_graph.csv'
            path_profile_graph_topk_csv = cur_graph_dir + '/profile_graph_topk.csv'
            path_readable_profile_graph_csv = cur_graph_dir + '/readable_profile_graph.csv'
            path_readable_profile_graph_topk_csv = cur_graph_dir + '/readable_profile_graph_topk.csv'
            pd.DataFrame(g_adj).to_csv(path_profile_graph_csv[:-4]+'_adj.csv',index=False)
            pd.DataFrame(g_std).to_csv(path_profile_graph_csv[:-4]+'_std.csv',index=False)
            cur_time_range = shift_time_range(cur_time_range,pd.Timedelta(seconds=graph_time_win),direction=1)


    def evolve_graph_sim_eval(self):
        graph_time_win = 3600
        whole_data = pd.concat([self.data.train,self.data.val,self.data.test])
        farmnodes = whole_data['instance'].unique()

        if self.args.dataset=='jlab':
            start_time = '2023-05-21 04:00:00'
            end_time = '2023-05-23 04:00:00'
        else:
            start_time = '2020-01-01 00:00:00'
            end_time = '2020-01-03 21:00:00'
        whole_data = slice_df_by_time_range(whole_data,[[start_time,end_time]])
        whole_data = whole_data.reset_index(drop=True)

        y_q75_instance = self.data.train.groupby('instance_id').quantile(0.75)[self.args.features]
        y_q25_instance = self.data.train.groupby('instance_id').quantile(0.25)[self.args.features]
        y_diff_instance = (y_q75_instance - y_q25_instance).sort_index()

        step = 1
        cur_time_range = [[start_time,shift_time_str(start_time, pd.Timedelta(seconds=graph_time_win), direction=1)]]

        graph_dir = os.path.join(os.path.dirname(self.args.paths['test_re']),'graph_evolve_sim_eval')
        Path(graph_dir+'/adj').mkdir(parents=True, exist_ok=True)
        Path(graph_dir+'/std').mkdir(parents=True, exist_ok=True)    
        result_df = pd.DataFrame()
        data = []
        for farmnode in farmnodes:
            prev_graph = None
            cur_time_range = [[start_time,shift_time_str(start_time, pd.Timedelta(seconds=graph_time_win), direction=1)]]
            while str2time(cur_time_range[0][1]) <= str2time(end_time):
                cur_data = slice_df_by_time_range(whole_data[whole_data['instance']==farmnode],cur_time_range)
                empty_row= cur_data.tail(1).copy()
                cur_data = pd.concat([cur_data,empty_row])
                if cur_data.shape[0] <= self.args.slide_win:
                    print(f'[WARN] Skip one time range for {farmnode}:  {cur_data.shape[0]} is less than slide win {self.args.slide_win}.')
                    cur_time_range = shift_time_range(cur_time_range,pd.Timedelta(seconds=graph_time_win),direction=1)
                    continue
                g_adj,g_std = profile_graph_4df(self.model, self.args, cur_data , self.data, y_diff_instance = y_diff_instance)
                cur_time_range_start_str = cur_time_range[0][0][8:13]
                np.save(graph_dir + f'/adj/{farmnode}_{cur_time_range_start_str}.npy', g_adj)
                np.save(graph_dir + f'/std/{farmnode}_{cur_time_range_start_str}.npy', g_std)
                if prev_graph is not None:
                    p_edges = pd.DataFrame(convert_adj2edges(prev_graph),columns=['source','destination','value'])
                    p_edges['source'] = p_edges['source'].apply(lambda x: int(x))
                    p_edges['destination'] = p_edges['destination'].apply(lambda x: int(x))
                    edges = pd.DataFrame(convert_adj2edges(g_adj),columns=['source','destination','value'])
                    edges['source'] = edges['source'].apply(lambda x: int(x))
                    edges['destination'] = edges['destination'].apply(lambda x: int(x))
                    data.append([farmnode,cur_time_range[0][0],get_similarity(p_edges,edges,distance='common')])
                prev_graph = g_adj
                cur_time_range = shift_time_range(cur_time_range,pd.Timedelta(seconds=graph_time_win),direction=1)
        result_df = pd.DataFrame(data,columns=['farmnode','timestamp','similarity'])
        result_df.to_csv(graph_dir + '/similarity.csv',index=False)

    def sensitity_graph_epsilon_eval(self):
        graph_time_win = 12
        epsilons = [1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
        # epsilons = [1e-2,1e-3]
        if self.args.dataset=='jlab':
            start_time = '2023-05-21 04:00:00'
            end_time = '2023-05-23 04:00:00'
            sample_interval = 60
        else:
            start_time = '2020-01-01 00:00:00'
            end_time = '2020-01-03 21:00:00'
            sample_interval = 10
        whole_data = pd.concat([self.data.train,self.data.val,self.data.test])
        farmnode = whole_data['instance'].unique()[0]
        whole_data = whole_data[whole_data['instance']==farmnode]
        whole_data = slice_df_by_time_range(whole_data,[[start_time,end_time]])
        whole_data = whole_data.reset_index(drop=True)
        data_12h = slice_df_by_time_range(whole_data,[[start_time,shift_time_str(start_time, pd.Timedelta(seconds=3600*graph_time_win), direction=1)]])
        data_tail = slice_df_by_time_range(whole_data,[[start_time,shift_time_str(start_time, pd.Timedelta(seconds=sample_interval), direction=1)]])
        cur_data = pd.concat([data_12h, data_tail], ignore_index=True)
        y_q75_instance_all = self.data.train.groupby('instance_id').quantile(0.75, numeric_only=True)[self.args.features]
        y_q25_instance_all = self.data.train.groupby('instance_id').quantile(0.25, numeric_only=True)[self.args.features]
        y_diff_instance_all = (y_q75_instance_all - y_q25_instance_all).sort_index()
        elapsed_time_list = []
        result_df = pd.DataFrame()
        data = []
        graph_dir = os.path.join(os.path.dirname(self.args.paths['train_re']),'graph_epsilon')

        epsilon = 1e-2
        cur_graph_dir = os.path.join(graph_dir, str(epsilon))
        Path(cur_graph_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(cur_graph_dir + f'/profile_graph_adj.npy'):
            self.args.trg_epsilon = epsilon
            start_time = time.time()
            g_adj_global,_ = profile_graph_4df(self.model, self.args, cur_data , self.data, y_diff_instance = y_diff_instance_all)
            elapsed_time = time.time() - start_time
            np.save(cur_graph_dir + f'/profile_graph_adj.npy', g_adj_global)
        else:
            g_adj_global = np.load(cur_graph_dir + f'/profile_graph_adj.npy')
        g_edges = pd.DataFrame(convert_adj2edges(g_adj_global),columns=['source','destination','value'])
        g_edges['source'] = g_edges['source'].apply(lambda x: int(x))
        g_edges['destination'] = g_edges['destination'].apply(lambda x: int(x))


        for epsilon in epsilons:
            cur_graph_dir = os.path.join(graph_dir, str(epsilon))
            Path(cur_graph_dir).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(cur_graph_dir + f'/profile_graph_adj.npy'):
                self.args.trg_epsilon = epsilon
                start_time = time.time()
                g_adj,_ = profile_graph_4df(self.model, self.args, cur_data , self.data, y_diff_instance = y_diff_instance_all)
                elapsed_time = time.time() - start_time
                np.save(cur_graph_dir + f'/profile_graph_adj.npy', g_adj)
            else:
                g_adj = np.load(cur_graph_dir + f'/profile_graph_adj.npy')
            edges = pd.DataFrame(convert_adj2edges(g_adj),columns=['source','destination','value'])
            edges['source'] = edges['source'].apply(lambda x: int(x))
            edges['destination'] = edges['destination'].apply(lambda x: int(x))

            data.append([epsilon,get_similarity(g_edges,edges,distance='common')])
        result_df = pd.DataFrame(data,columns=['epsilon','similarity'])
        result_df.to_csv(graph_dir + '/epsilon_similarity.csv',index=False)
        print(result_df.to_markdown())


    def measure_graph_time(self):
        graph_time_wins = [1,12,24,24*7]
        num_nodes = [1]
        if self.args.dataset=='jlab':
            start_time = '2023-05-21 04:00:00'
            end_time = '2023-05-23 04:00:00'
            sample_interval = 60
        else:
            start_time = '2020-01-01 00:00:00'
            end_time = '2020-01-03 21:00:00'
            sample_interval = 10
        whole_data = pd.concat([self.data.train,self.data.val,self.data.test])
        farmnode = whole_data['instance'].unique()[0]
        whole_data = whole_data[whole_data['instance']==farmnode]
        whole_data = slice_df_by_time_range(whole_data,[[start_time,end_time]])
        whole_data = whole_data.reset_index(drop=True)
        data_1h = slice_df_by_time_range(whole_data,[[start_time,shift_time_str(start_time, pd.Timedelta(seconds=3600), direction=1)]])
        data_tail = slice_df_by_time_range(whole_data,[[start_time,shift_time_str(start_time, pd.Timedelta(seconds=sample_interval), direction=1)]])
        y_q75_instance_all = self.data.train.groupby('instance_id').quantile(0.75, numeric_only=True)[self.args.features]
        y_q25_instance_all = self.data.train.groupby('instance_id').quantile(0.25, numeric_only=True)[self.args.features]
        y_diff_instance_all = (y_q75_instance_all - y_q25_instance_all).sort_index()
        elapsed_time_list = []
        for num_node in num_nodes:
            for graph_time_win in graph_time_wins:
                print(f'----Time win: {graph_time_win}h, # Nodes: {num_node} ----')
                repeated_data = [data_1h.copy() for _ in range(graph_time_win*num_node)]
                cur_data = pd.concat(repeated_data + [data_tail], ignore_index=True)
                start_time = time.time()
                y_q75_instance = cur_data.groupby('instance_id').quantile(0.75, numeric_only=True)[self.args.features]
                y_q25_instance = cur_data.groupby('instance_id').quantile(0.25, numeric_only=True)[self.args.features]
                y_diff_instance = (y_q75_instance - y_q25_instance).sort_index()
                y_diff_instance_all.update(y_diff_instance)
                g_adj,g_std = profile_graph_4df(self.model, self.args, cur_data , self.data, y_diff_instance = y_diff_instance_all)
                elapsed_time = time.time() - start_time
                print('Elapsed time:', elapsed_time,'s')
                print(f'----Time win: {graph_time_win}h, # Nodes: 100 ----')
                print('Elapsed time:', elapsed_time,'s')
                elapsed_time_list.append(elapsed_time)

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
        
        self.data.train_gt = self.data.train.copy()
        self.data.test_gt = self.data.test.copy()
        self.data.val_gt = self.data.val.copy()

        self.data.train = align_df(self.data.train_re, self.data.train, self.data.train_dataset.rang, self.args)
        self.data.test = align_df(self.data.test_re, self.data.test, self.data.test_dataset.rang, self.args)
        self.data.val = align_df(self.data.val_re, self.data.val, self.data.val_dataset.rang, self.args)

        if not (hasattr(self.args, 'load_model_path') and  self.args.load_model_path != '' and self.args.reload_args):
            save_args(self.args)

        if self.args.store:
            self.args.paths['profile_graph_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/profile_graph.csv'
            self.args.paths['readable_profile_graph_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/readable_profile_graph.csv'

            self.args.paths['profile_graph_topk_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/profile_graph_topk.csv'
            self.args.paths['readable_profile_graph_topk_csv'] = os.path.dirname(self.args.paths['graph_csv']) + '/readable_profile_graph_topk.csv'
            if not os.path.exists(self.args.paths['profile_graph_csv']):
                print('>>>>>>>>> Sampling Graph')
                self.profile_graph_adj, self.profile_graph_adj_std = self.profile_graph_wrapper(self.model, self.args, self.data.val_dataloader,self.data)
                pd.DataFrame(self.profile_graph_adj).to_csv(self.args.paths['profile_graph_csv'][:-4]+'_adj.csv',index=False)
                pd.DataFrame(self.profile_graph_adj_std).to_csv(self.args.paths['profile_graph_csv'][:-4]+'_adj_std.csv',index=False)
                learned_graph = convert_adj2edges_wrapper(self.profile_graph_adj,len(self.args.features),dim=0, norm_flag=False)
                save_graph(learned_graph,self.args.features, self.args.paths['profile_graph_csv'] ,self.args.paths['readable_profile_graph_csv'])
                print('save graph to {}'.format(self.args.paths['profile_graph_csv']))
        regression_score.get_score(self.data, self.args, draw_eval=True,save2disk=self.args.store,data_filter=self.args.data_filter)
        g_extracter = Graph_extracter(self.model,self.args,self.data)
        if self.args.dtask == 'anomaly':
            anomaly_score.get_score(self.data, self.args, graph_path=self.args.paths['profile_graph_topk_csv'], draw_eval=True,save2disk=self.args.store, pCache=self.args.pCache,data_filter=self.args.data_filter,g_extracter=g_extracter) # have bug for no_store parameter

    def load(self):
        if not hasattr(self.args, 'load_model_path') or self.args.load_model_path == '':
            return True
        else:
            self.model.load_state_dict(torch.load(self.args.paths['best_pt'], map_location=torch.device(self.args.device)))