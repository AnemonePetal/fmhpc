import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from util.postprocess import align_df_2d
from util.save import save_result_2d
from util.Data import filter_df
from util.argparser import save_args
from util.data_loader import set_dataloader
import optuna
from util.env import *
from util.deterministic import set_deterministic
import math
from util.anomaly_dtask import anomaly_score
from util import regression_score
import time

class TimeMixerLayer(nn.Module):
    def __init__(self, width_time: int, dropout: float):
        super(TimeMixerLayer, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=[width_time])

        self.lin = nn.Linear(in_features=width_time, out_features=width_time)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x.transpose(2, 3)).transpose(2, 3)
        y = torch.transpose(y, 2, 3)
        y = self.lin(y)
        y = self.act(y)
        y = torch.transpose(y, 2, 3)
        y = self.dropout(y)
        return x + y

class FeatMixerLayer(nn.Module):
    def __init__(self, width_feats: int, width_feats_hidden: int, dropout: float):
        super(FeatMixerLayer, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=[width_feats])
        
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

class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.full_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.full_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.full_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, indices=None, jobs=None, dim = 1):
        batch= indices.shape[0]
        num_hnodes = indices.shape[1]
        time_win = indices.shape[2]
        weight = self.full_weight.index_select(1, indices.reshape(-1))
        weight = weight.reshape(self.out_features, batch, num_hnodes, time_win)


        y = torch.einsum('bijk,lbki->bijl', input, weight)
        return y

class NodeMixerLayer(nn.Module):
    def __init__(self, num_hnodes: int , width_batch: int,  width_hnodes_hidden: int, width_hnodes_group: int,dropout: float):
        super(NodeMixerLayer, self).__init__()
        self.width_batch = width_batch
        self.num_hnodes = num_hnodes
        self.norm = nn.LayerNorm(normalized_shape=[num_hnodes])
        self.lin_1 = MaskLinear(in_features=num_hnodes, out_features=width_hnodes_group)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, entryid: torch.Tensor) -> torch.Tensor:
        y = self.norm(x.transpose(1, 3)).transpose(1, 3)
        
        batch_size = y.shape[0]
        y = x.permute(0, 2, 3, 1)

        y = self.lin_1(y, indices= entryid)
        y = self.act(y)
        y = self.dropout_1(y)

        y = y.permute(0, 3, 1, 2)

        return x + y

class Mixer(nn.Module):
    def __init__(self, input_length: int, forecast_length: int, num_feats: int, feat_mixing_hidden_channels: int, num_mixer_layers: int, num_hnodes: int, num_batch: int , width_hnodes_group:int,  dropout: float, eps: float = 1e-8, tb_writer: SummaryWriter = None):
        super(Mixer, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_feats))
        self.shift = nn.Parameter(torch.zeros(num_feats))

        self.num_batch = num_batch
        self.width_hnodes_group = width_hnodes_group
        self.num_feats = num_feats
        self.writer= tb_writer

        self.temp_proj = nn.Linear(in_features=input_length, out_features=forecast_length)
        mixer_layers = []
        for _ in range(num_mixer_layers):
            mixer_layers.append(NodeMixerLayer(num_hnodes= num_hnodes, width_batch=num_batch, width_hnodes_hidden=feat_mixing_hidden_channels, width_hnodes_group=width_hnodes_group,dropout=dropout))
            mixer_layers.append(TimeMixerLayer(width_time=input_length, dropout=dropout))
            mixer_layers.append(FeatMixerLayer(width_feats=num_feats, width_feats_hidden=feat_mixing_hidden_channels, dropout=dropout))
        self.mixer_layers = nn.ModuleList(mixer_layers)

    def forward(self, x: torch.Tensor, entryid: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 2)
        mean = torch.mean(x, dim=2, keepdim=True)
        var = torch.var(x, dim=2, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.scale + self.shift
        
        for i, mixer_layer in enumerate(self.mixer_layers):
            if i%3 == 0:
                x = mixer_layer(x,entryid)
            else:
                x = mixer_layer(x)
        x = torch.transpose(x, 2, 3)
        x = self.temp_proj(x)
        x = torch.transpose(x, 2, 3)
        x = (x - self.shift) / self.scale
        x = x * torch.sqrt(var + self.eps) + mean
        return x
    
    def generate_representation(self, x: torch.Tensor, entryid: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.scale + self.shift
        
        for i, mixer_layer in enumerate(self.mixer_layers):
            if i%3 == 2:
                x = mixer_layer(x,entryid)
            else:
                x = mixer_layer(x)
        x = x.permute(0, 2, 1)
        return x

class FM_NM_wrapper(object,):
    def __init__(self, args,data,tensorboard=False):
        if tensorboard and args.store and args.load_model_path=='':
            self.writer = SummaryWriter(os.path.dirname(args.paths['test_re'])+'/tensorboard')
        else:
            self.writer = None
        if args.deterministic:
            set_deterministic(args)
        set_dataloader(data,args)
        self.args= args
        self.data = data
        self._setup_model()
    
    def _setup_model(self):
        self.model = Mixer(
                        input_length=self.args.slide_win, 
                        forecast_length=self.args.pred_win, 
                        num_feats= len(self.args.features), 
                        feat_mixing_hidden_channels= self.args.out_layer_inter_dim, 
                        num_mixer_layers=self.args.out_layer_num, 
                        num_hnodes = len(self.data.train['instance'].unique()),
                        num_batch = self.args.batch,
                        width_hnodes_group = self.args.hnodes_size,
                        dropout=0.2,
                        tb_writer= self.writer,
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
            
            micro_batch_size = self.args.micro_batch
            
            for x, y, labels, edge_index, timestamp, instance_id, status_mask, job_group, job_id,_ in dataloader:
                batch_size = x.size(0)
                
                if batch_size <= micro_batch_size:
                    micro_x = x.float().to(self.args.device)
                    micro_y = y.float().to(self.args.device)
                    micro_instance_id = instance_id.int().to(self.args.device)
                    
                    predictions = self.model(micro_x, micro_instance_id)
                    
                    loss = loss_func(predictions, micro_y).cpu().item()
                    batch_loss = loss
                    
                    predictions_np = predictions.cpu().numpy()
                    predictions_np = predictions_np.reshape(-1, predictions_np.shape[3])
                    test_predicted_list.append(predictions_np)
                    
                    del predictions
                    del micro_x
                    del micro_y
                    del micro_instance_id
                    torch.cuda.empty_cache()
                else:
                    num_micro_batches = math.ceil(batch_size / micro_batch_size)
                    batch_loss = 0.0
                    
                    for i in range(num_micro_batches):
                        start_idx = i * micro_batch_size
                        end_idx = min((i + 1) * micro_batch_size, batch_size)
                        
                        micro_x = x[start_idx:end_idx].float().to(self.args.device)
                        micro_y = y[start_idx:end_idx].float().to(self.args.device)
                        micro_instance_id = instance_id[start_idx:end_idx].int().to(self.args.device)
                        
                        predictions = self.model(micro_x, micro_instance_id)
                        
                        loss = loss_func(predictions, micro_y).cpu().item()
                        batch_loss += loss * (end_idx - start_idx) / batch_size
                        
                        predictions_np = predictions.cpu().numpy()
                        predictions_np = predictions_np.reshape(-1, predictions_np.shape[3])
                        test_predicted_list.append(predictions_np)
                        
                        del predictions
                        del micro_x
                        del micro_y
                        del micro_instance_id
                        torch.cuda.empty_cache()
                
                test_loss.append(batch_loss)
                
            test_predicted_list = np.concatenate(test_predicted_list)
            test_loss = np.mean(test_loss)
        return test_loss, test_predicted_list

    def train(self):
        def log_gradient_norms(model, writer, step):
            """Log L2 norm of gradients per layer to TensorBoard"""
            layer_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if '.' in name:
                        layer_type = name.split('.')[0]
                        param_norm = param.grad.norm(2).item()
                        
                        if layer_type in layer_norms:
                            layer_norms[layer_type].append(param_norm)
                        else:
                            layer_norms[layer_type] = [param_norm]
            
            for layer_type, norms in layer_norms.items():
                avg_norm = sum(norms) / len(norms)
                writer.add_scalar(f'gradient_norm/{layer_type}', avg_norm, step)
            
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('gradient_norm/total', total_norm, step)
        print("Training FM_NM...")
        train_loss_list = []
        epoch_times = []
        min_loss = float('inf')
        best_model_state = None
        stop_improve_count = 0
        
        micro_batch_size = self.args.micro_batch

        for step in range(self.args.epoch):
            self.model.train()
            train_loss = []
            start_time = time.time()
            
            for x, y, labels, edge_index, timestamp, instance_id, status_mask, job_group, job_id,_ in self.data.train_dataloader:
                batch_size = x.size(0)
                
                if batch_size <= micro_batch_size:
                    self.optimizer.zero_grad()
                    
                    micro_x = x.float().to(self.args.device)
                    micro_y = y.float().to(self.args.device)
                    micro_instance_id = instance_id.int().to(self.args.device)
                    
                    predictions = self.model(micro_x, micro_instance_id)
                    
                    loss = F.mse_loss(predictions, micro_y, reduction='mean')
                    
                    loss.backward()
                    
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    train_loss.append(loss.item())
                    
                    del micro_x
                    del micro_y
                    del micro_instance_id
                    del predictions
                    del loss
                else:
                    num_micro_batches = math.ceil(batch_size / micro_batch_size)
                    accumulated_loss = 0
                    
                    self.optimizer.zero_grad()
                    
                    for i in range(num_micro_batches):
                        start_idx = i * micro_batch_size
                        end_idx = min((i + 1) * micro_batch_size, batch_size)
                        
                        micro_x = x[start_idx:end_idx].float().to(self.args.device)
                        micro_y = y[start_idx:end_idx].float().to(self.args.device)
                        micro_instance_id = instance_id[start_idx:end_idx].int().to(self.args.device)
                        
                        predictions = self.model(micro_x, micro_instance_id)
                        
                        loss = F.mse_loss(predictions, micro_y, reduction='mean') / num_micro_batches
                        
                        loss.backward()
                        
                        accumulated_loss += loss.item() * num_micro_batches * (end_idx - start_idx) / batch_size
                        
                        del micro_x
                        del micro_y
                        del micro_instance_id
                        del predictions
                        del loss
                    
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    train_loss.append(accumulated_loss)
            
            mean_loss = np.mean(train_loss)
            train_loss_list.append(mean_loss)
            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            print('epoch: {}, loss: {}, lapsed time: {}'.format(step, mean_loss, epoch_time))
            
            if self.writer is not None:
                self.writer.add_scalar('total_loss', mean_loss, step)
                log_gradient_norms(self.model, self.writer, step)

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

        self.model.load_state_dict(torch.load(self.args.paths['cache_pt']))
        if self.args.store:
            if os.path.exists(self.args.paths['cache_pt']):
                os.rename(self.args.paths['cache_pt'], self.args.paths['best_pt'])
            loss_df = pd.DataFrame(train_loss_list, columns=['loss'])
            loss_df.to_csv(self.args.paths['loss_csv'], index=False)
        average_time_per_epoch = np.mean(epoch_times)
        print('> Average time per epoch: %.3f seconds' % average_time_per_epoch)

        return train_loss_list, best_model_state
    
    def score(self):
        if not os.path.exists(self.args.paths['train_re']):
            _, self.data.train_re = self.predict(self.data.train_dataloader)
            self.data.train_re = save_result_2d(self.args, self.data, result_type='train',save2disk=self.args.pCache)
        else:
            self.data.train_re = pd.read_csv(self.args.paths['train_re'])
            self.data.train_re['timestamp'] = pd.to_datetime(self.data.train_re['timestamp'])
            self.data.train_re = filter_df(self.data.train_re, self.args.data_filter)
        if not os.path.exists(self.args.paths['test_re']):
            _, self.data.test_re = self.predict(self.data.test_dataloader)
            self.data.test_re = save_result_2d(self.args, self.data, result_type='test',save2disk=self.args.pCache)
        else:
            self.data.test_re = pd.read_csv(self.args.paths['test_re'])
            self.data.test_re['timestamp'] = pd.to_datetime(self.data.test_re['timestamp'])
            self.data.test_re = filter_df(self.data.test_re, self.args.data_filter)

        if not os.path.exists(self.args.paths['val_re']):
            _, self.data.val_re = self.predict(self.data.val_dataloader)
            self.data.val_re = save_result_2d(self.args, self.data, result_type='val',save2disk=self.args.pCache)
        else:
            self.data.val_re = pd.read_csv(self.args.paths['val_re'])
            self.data.val_re['timestamp'] = pd.to_datetime(self.data.val_re['timestamp'])
            self.data.val_re = filter_df(self.data.val_re, self.args.data_filter)
    
        self.data.train_gt = self.data.train.copy()
        self.data.test_gt = self.data.test.copy()
        self.data.val_gt = self.data.val.copy()

        self.data.train = align_df_2d(self.data.train_re, self.data.train, self.data.train_dataset.rang, self.data.train_dataset.rang_hnode, self.args)
        self.data.test = align_df_2d(self.data.test_re, self.data.test, self.data.test_dataset.rang, self.data.test_dataset.rang_hnode, self.args)
        self.data.val = align_df_2d(self.data.val_re, self.data.val, self.data.val_dataset.rang, self.data.val_dataset.rang_hnode, self.args)
        
        if not (hasattr(self.args, 'load_model_path') and  self.args.load_model_path != '' and self.args.reload_args):
            save_args(self.args)

        regression_score.get_score(self.data, self.args, draw_eval=True,save2disk=self.args.store,data_filter=self.args.data_filter)
    
    def load(self):
        if not hasattr(self.args, 'load_model_path') or self.args.load_model_path == '':
            return True
        else:

            self.model.load_state_dict(torch.load(self.args.paths['best_pt']))