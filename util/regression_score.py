from util.data_stat import *
import numpy as np
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, roc_curve
from sklearn import metrics
from util.plot import *
from util.Data import filter_df
from copy import deepcopy
import os
from statsmodels.distributions.empirical_distribution import ECDF
import torch
import json
import ast
from sklearn.preprocessing import RobustScaler
from pathlib import Path

def get_score(data, args, draw_eval=True, draw_stat=False,verbose=True, save2disk=False,data_filter={}):

    if verbose:
        print('===** Regression **===')
    get_feature_scores(data.test_re,data.test, data.val_re,data.val,data.train_re, data.train, args.features, args, metric='r2', draw_eval=draw_eval, verbose=False, save2disk=save2disk,savepath=args.paths['score_dir']+'regression/')
    get_feature_scores(data.test_re,data.test, data.val_re,data.val,data.train_re, data.train, args.features, args, metric='r2', inverse_scaler=data.scaler, draw_eval=draw_eval, verbose=False, save2disk=save2disk,savepath=args.paths['score_dir']+'regression/')
    get_feature_scores(data.test_re,data.test, data.val_re,data.val,data.train_re, data.train, args.features, args, metric='rmse',draw_eval=draw_eval, verbose=verbose, save2disk=save2disk, savepath=args.paths['score_dir']+'regression/')
    get_feature_scores(data.test_re,data.test, data.val_re,data.val,data.train_re, data.train, args.features, args, metric='rmse', inverse_scaler=data.scaler,draw_eval=draw_eval, verbose=False, save2disk=save2disk, savepath=args.paths['score_dir']+'/regression/')

def get_feature_scores(test_re,test_gt, val_re,val_gt,train_re, train_gt, feature_map,args, metric='mse',draw_eval=True, verbose=True, save2disk=False,inverse_scaler=None,savepath='./'):
    inverse_scaler_str = 'normalized' if inverse_scaler is None else 'original'
    if os.path.exists(savepath+ inverse_scaler_str + "_metrics_"+metric+".csv"):
        feature_metric = pd.read_csv(savepath + inverse_scaler_str + "_metrics_"+metric+".csv")
        if verbose:
            print('======** {0} {1} **======'.format(inverse_scaler_str, metric))
            for data_type in ['train','val','test']:
                stat_metric = np.mean(feature_metric[data_type])
                print(f'{data_type} : {stat_metric}')

        return feature_metric

    if inverse_scaler is not None:
        train_re = train_re.copy()
        train_gt = train_gt.copy()
        val_re = val_re.copy()
        val_gt = val_gt.copy()
        test_re = test_re.copy()
        test_gt = test_gt.copy()

        train_re= inverse_scaler.inverse(train_re,feature_map)
        train_gt= inverse_scaler.inverse(train_gt,feature_map)
        val_re= inverse_scaler.inverse(val_re,feature_map)
        val_gt= inverse_scaler.inverse(val_gt,feature_map)
        test_re= inverse_scaler.inverse(test_re,feature_map)
        test_gt= inverse_scaler.inverse(test_gt,feature_map)


    feature_metric = pd.DataFrame()
    for data_type in ['train','val','test']:
        if data_type == 'train':
            df_re = train_re
            df_gt = train_gt
        elif data_type == 'val':
            df_re = val_re
            df_gt = val_gt
        elif data_type == 'test':
            df_re = test_re
            df_gt = test_gt
        else:
            raise ValueError('data_type should be train or val or test')
        row = []
        for feature in feature_map:
            if metric == 'mse':
                row.append(metrics.mean_squared_error(df_gt[feature], df_re[feature]))
            elif metric == 'mae':
                row.append(metrics.mean_absolute_error(df_gt[feature], df_re[feature]))
            elif metric == 'r2':
                r2_score = metrics.r2_score(df_gt[feature], df_re[feature])
                if r2_score == 0:
                    r2_score = np.nan
                row.append(r2_score)
            elif metric == 'rmse':
                row.append(np.sqrt(metrics.mean_squared_error(df_gt[feature], df_re[feature])))
            elif metric == 'mape':
                frac_list = np.abs((df_gt[feature] - df_re[feature]) / df_gt[feature])
                if np.isinf(frac_list).any():
                    row.append(np.nan)
                else:
                    row.append(np.mean(frac_list))
        feature_metric[data_type] = row
    
    feature_metric.index = feature_map
    if save2disk:
        Path(savepath).mkdir(parents=True, exist_ok=True)
        feature_metric.to_csv(savepath + inverse_scaler_str + "_metrics_"+metric+".csv",index=True)
    if verbose:
        print('======** {0} {1} **======\n'.format(inverse_scaler_str, metric))
        for data_type in ['train','val','test']:
            stat_metric = np.mean(feature_metric[data_type])
            print(f'{data_type} : {stat_metric}')

    return feature_metric

def feature_stat(test_re,test_gt, val_re,val_gt,train_re, train_gt, feature_map,args, draw_stat=True, verbose=True, save2disk=False,savepath='train_scores_l_csv',inverse_scaler=None):
    cur_dir = "/".join(args.paths[savepath].split("/")[:-1]) + "/feature_stat"

    if not os.path.exists(cur_dir):
        return
    elif os.path.exists(cur_dir+"/re_stat.csv"):
        return 
    
    if inverse_scaler is not None:
        train_re= inverse_scaler.inverse(train_re,feature_map)
        train_gt= inverse_scaler.inverse(train_gt,feature_map)
        val_re= inverse_scaler.inverse(val_re,feature_map)
        val_gt= inverse_scaler.inverse(val_gt,feature_map)
        test_re= inverse_scaler.inverse(test_re,feature_map)
        test_gt= inverse_scaler.inverse(test_gt,feature_map)

    re = pd.concat([train_re,val_re,test_re])
    gt = pd.concat([train_gt,val_gt,test_gt])
    

    def plot_line(re_df,gt_df,x,y,xlabel,ylabel,path):
        dir_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fig = go.Figure()
        if isinstance(y,list):
            visible_list = [False]*len(y)*2
            button_list = []
            for i in range(len(y)):
                fig.add_scattergl(x=re_df[x],y=re_df[y[i]],name= 'ground truth: '+ y[i])
                fig.add_scattergl(x=gt_df[x],y=gt_df[y[i]],name= 'prediction: '+ y[i])
                visible_list[i*2:i*2+2] = [True]*2
                button_list.append(dict(label=y[i],
                                        method="update",
                                        args=[{"visible": deepcopy(visible_list)},
                                            {"title": y[i]}]))
                visible_list[i*2:i*2+2] = [False]*2
        else:        
            raise ValueError('y should be a list')
        fig.update_layout(updatemenus=[dict(active=0, buttons=button_list)])
        fig.update_layout(
            title= ylabel[1],
            xaxis_title=xlabel,
            yaxis_title=ylabel[1]+':'+ylabel[0],
        )
        fig.write_html(path)

    def plot_autocorr(re_series,gt_series,title,path):
        dir_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fig = go.Figure()
        re_autocorr = []
        gt_autocorr = []
        for i in range(1,200):
            re_autocorr.append(re_series.autocorr(lag=i))
            gt_autocorr.append(gt_series.autocorr(lag=i))
        fig.add_scattergl(x=list(range(1,200)),y=gt_autocorr,name= 'ground truth')
        fig.add_scattergl(x=list(range(1,200)),y=re_autocorr,name= 'prediction')
        fig.update_layout(
            title= title,
            xaxis_title="lag",
            yaxis_title="autocorrelation",
        )
        fig.write_html(path)

    def plot_hist(re_series,gt_series,title,path):
        dir_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        colors = px.colors.qualitative.Plotly

        mean_re = re_series.mean()
        mean_gt = gt_series.mean()
        std_re = re_series.std()
        std_gt = gt_series.std()
        
        fig = go.Figure()
        fig.add_histogram(x=gt_series,name= 'ground truth',marker_color=colors[0], legendgroup='ground truth',autobinx=False)
        fig.add_histogram(x=re_series,name= 'prediction',marker_color=colors[1], legendgroup='prediction',autobinx=False)
        full_fig = fig.full_figure_for_development(warn=False)
        min_value = full_fig['layout']['yaxis']['range'][0]
        max_value = full_fig['layout']['yaxis']['range'][1]
        fig.add_trace(go.Scatter(x=[mean_gt, mean_gt], y=[min_value,max_value], mode='lines', name='mean: {:.3f}'.format(mean_gt), line=dict(color=colors[0], dash='dot'), legendgroup='ground truth'))
        fig.add_trace(go.Scatter(x=[mean_gt-std_gt, mean_gt-std_gt], y=[min_value,max_value], mode='lines', name='mean-std({:.3f})'.format(std_gt), line=dict(color=colors[0], dash='dash'), legendgroup='ground truth'))
        fig.add_trace(go.Scatter(x=[mean_gt+std_gt, mean_gt+std_gt], y=[min_value,max_value], mode='lines', name='mean+std({:.3f})'.format(std_gt), line=dict(color=colors[0], dash='dash'), legendgroup='ground truth'))

        fig.add_trace(go.Scatter(x=[mean_re, mean_re], y=[min_value,max_value], mode='lines', name='mean: {:.3f}'.format(mean_re), line=dict(color=colors[1], dash='dot'), legendgroup='prediction'))
        fig.add_trace(go.Scatter(x=[mean_re-std_re, mean_re-std_re], y=[min_value,max_value], mode='lines', name='mean-std({:.3f})'.format(std_re), line=dict(color=colors[1], dash='dash'), legendgroup='prediction'))
        fig.add_trace(go.Scatter(x=[mean_re+std_re, mean_re+std_re], y=[min_value,max_value], mode='lines', name='mean+std({:.3f})'.format(std_re), line=dict(color=colors[1], dash='dash'), legendgroup='prediction'))

        fig.update_layout(
            title= title,
            xaxis_title="value",
            yaxis_title="count",
        )
        fig.write_html(path)

    def plot_cdf(re_series,gt_series, title,path):
        dir_path = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        colors = px.colors.qualitative.Plotly

        feb_gt_ecdf = ECDF(gt_series)
        feb_re_ecdf = ECDF(re_series)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feb_gt_ecdf.x, y=feb_gt_ecdf.y,name= 'ground truth',marker_color=colors[0], legendgroup='ground truth'))
        fig.add_trace(go.Scatter(x=feb_re_ecdf.x, y=feb_re_ecdf.y,name= 'prediction',marker_color=colors[1], legendgroup='prediction'))
        fig.update_layout(
            title= title,
            xaxis_title="value",
            yaxis_title="cumulative probability",
        )
        fig.write_html(path)



    re_stat = pd.DataFrame(columns=['feature','instance','mean','variance','standard_deviation','coefficient_of_variation','median','max','min'])
    gt_stat = pd.DataFrame(columns=['feature','instance','mean','variance','standard_deviation','coefficient_of_variation','median','max','min'])

    instances= re.instance.unique()
    for i,feature in enumerate(feature_map):
        for instance in instances:
            re_one = re[re['instance']==instance][['timestamp']+[feature]].copy()
            gt_one = gt[gt['instance']==instance][['timestamp']+[feature]].copy()            
            len_csv_pd = re_one.shape[0]
            re_one['mean'] = re_one[feature].rolling(window=len_csv_pd,min_periods=1).mean()
            re_one['variance'] = re_one[feature].rolling(window=len_csv_pd,min_periods=1).var()
            re_one['standard_deviation'] = re_one[feature].rolling(window=len_csv_pd,min_periods=1).std()
            re_one['coefficient_of_variation'] = re_one['standard_deviation']/re_one['mean']
            re_one['median'] = re_one[feature].rolling(window=len_csv_pd,min_periods=1).median()
            re_one['max'] = re_one[feature].rolling(window=len_csv_pd,min_periods=1).max()
            re_one['min'] = re_one[feature].rolling(window=len_csv_pd,min_periods=1).min()

            len_csv_pd = gt_one.shape[0]
            gt_one['mean'] = gt_one[feature].rolling(window=len_csv_pd,min_periods=1).mean()
            gt_one['variance'] = gt_one[feature].rolling(window=len_csv_pd,min_periods=1).var()
            gt_one['standard_deviation'] = gt_one[feature].rolling(window=len_csv_pd,min_periods=1).std()
            gt_one['coefficient_of_variation'] = gt_one['standard_deviation']/gt_one['mean']
            gt_one['median'] = gt_one[feature].rolling(window=len_csv_pd,min_periods=1).median()
            gt_one['max'] = gt_one[feature].rolling(window=len_csv_pd,min_periods=1).max()
            gt_one['min'] = gt_one[feature].rolling(window=len_csv_pd,min_periods=1).min()
            
            y_list = ['mean','variance','standard_deviation','coefficient_of_variation','median','max','min']
            re_stat.loc[re_stat.shape[0]] = [feature,instance]+[re_one[i].iloc[-1] for i in y_list]
            gt_stat.loc[gt_stat.shape[0]] = [feature,instance]+[gt_one[i].iloc[-1] for i in y_list]
        
        if draw_stat:
            plot_autocorr(re[feature],gt[feature],title=feature+" autocorrelation",path=cur_dir+ "/"+feature+"/"+"allinstance_"+feature+"_autocorr.html")
            plot_hist(re[feature],gt[feature],title=feature+" histogram",path=cur_dir+ "/"+feature+"/"+"allinstance_"+feature+"_hist.html")
            plot_cdf(re[feature],gt[feature],title=feature+" cdf",path=cur_dir+ "/"+feature+"/"+"allinstance_"+feature+"_cdf.html")
    re_stat_byfeature = re_stat.groupby(['feature']).mean(numeric_only=True).reset_index()
    gt_stat_byfeature = gt_stat.groupby(['feature']).mean(numeric_only=True).reset_index()

    if save2disk:
        re_stat_byfeature.to_csv(cur_dir+"/re_stat.csv",index=False)
        gt_stat_byfeature.to_csv(cur_dir+"/gt_stat.csv",index=False)
