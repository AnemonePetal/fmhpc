from util.data_stat import *
import numpy as np
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, roc_curve
from sklearn import metrics
from util.plot import *
from copy import deepcopy
import os
from statsmodels.distributions.empirical_distribution import ECDF
import torch
import json
import ast
from sklearn.preprocessing import RobustScaler
from util.save import dtask_path
from util.postprocess import check_consisitency, sync_sort
from util.net_prep import convert_adj2edges_wrapper,convert_adj2edges
from util.extract_graph import g2df,get_similarity

def get_score(data, args, graph_path,draw_eval=True, draw_stat=False,verbose=True, save2disk=False, pCache=True ,data_filter={},g_extracter=None):
    gt_labels = [data.train_re['label'].values.tolist(),data.val_re['label'].values.tolist(),data.test_re['label'].values.tolist()]
    result_dir = 'anomaly_detect({})'.format(graph_path.split('/')[-1].split('.')[0])
    dtask_paths = dtask_path(args,result_dir,build_dir=save2disk)
    if verbose:
        print('=========================** Anomaly **============================\n')
    train_scores, val_scores, test_scores = get_full_error_scores(args, data.test_re, data.test, data.val_re, data.val, data.train_re, data.train,args.features,savepaths=dtask_paths, save2disk=save2disk, pCache=pCache ,by='instance')    
    train_scores, data.train = sync_sort(train_scores, data.train, by_column=["timestamp","instance"])
    val_scores, data.val = sync_sort(val_scores, data.val, by_column=["timestamp","instance"])
    test_scores, data.test = sync_sort(test_scores, data.test, by_column=["timestamp","instance"])

    info_l = get_val_performance_data(args, test_scores, val_scores,train_scores, gt_labels, args.features, graph_path,savepaths=dtask_paths, save2disk=save2disk, pCache=pCache, draw_eval=draw_eval, verbose= verbose,threshold_per=args.threshold_per,g_extracter=g_extracter)

    return info_l

def get_full_error_scores(args, test_re, test_gt, val_re, val_gt, train_re, train_gt, feature_map, savepaths ,save2disk=True, pCache=True,by=None,data_filter={}):
    if  not os.path.exists(savepaths['train_scores']) or not os.path.exists(savepaths['val_scores']) or not os.path.exists(savepaths['test_scores']):

        train_scores = deepcopy(train_re.iloc[:,~train_re.columns.isin(args.features)])
        val_scores = deepcopy(val_re.iloc[:,~val_re.columns.isin(args.features)])
        test_scores = deepcopy(test_re.iloc[:,~test_re.columns.isin(args.features)])
        
        if 'GPU' in test_gt.columns and 'GPU' not in test_scores.columns:
            test_scores['GPU'] = test_gt['GPU']

        train_scores[feature_map] = get_deviation(train_re[feature_map],train_gt[feature_map])
        val_scores[feature_map] = get_deviation(val_re[feature_map],val_gt[feature_map])
        test_scores[feature_map] = get_deviation(test_re[feature_map],test_gt[feature_map])
        
        train_scores['source_flag']='train'
        val_scores['source_flag']='val'
        test_scores['source_flag']='test'
        if args.dataset == 'olcfcutsec':
            whole_data = test_scores
        else:
            whole_data = pd.concat([train_scores,val_scores,test_scores])
        whole_data = whole_data.sort_values(by='timestamp').reset_index(drop=True)
        
        if args.dataset == 'olcfcutsec':
            avg_len = (whole_data.shape[0]/len(whole_data.instance.unique()))
            min_period_per = 60/avg_len
            del test_scores
        else:
            min_period_per = train_scores.shape[0]/whole_data.shape[0]
            del train_scores,val_scores,test_scores
        if min_period_per >1:
            raise ValueError('The min_period_per is larger than 1, please check the data')

        def groupby_helper(group,min_period_per):
            epsilon=1e-2
            if min_period_per ==1:
                median = group.median()
                q75 = group.quantile(0.75)
                q25 = group.quantile(0.25)
            else:
                min_period = int(min_period_per*group.shape[0])
                median = group.expanding(min_periods = min_period).median()
                q75 = group.expanding(min_periods = min_period).quantile(0.75)
                q25 = group.expanding(min_periods = min_period).quantile(0.25)
                median.iloc[:min_period] = median.iloc[min_period]
                q75.iloc[:min_period] = q75.iloc[min_period]
                q25.iloc[:min_period] = q25.iloc[min_period]
            return abs(group - median) / (q75 - q25 + epsilon)

        if by is None:
            whole_data[feature_map] = whole_data[feature_map].apply(groupby_helper,min_period_per=min_period_per)
        else:
            whole_data[feature_map] = whole_data.groupby(by, group_keys=False)[feature_map].apply(groupby_helper,min_period_per=min_period_per)

        if args.dataset == 'olcfcutsec':
            if by is None:
                train_scores[feature_map] = train_scores[feature_map].apply(groupby_helper,min_period_per=1)
                val_scores[feature_map] = val_scores[feature_map].apply(groupby_helper,min_period_per=1)
            else:
                train_scores[feature_map] = train_scores.groupby(by, group_keys=False)[feature_map].apply(groupby_helper,min_period_per=1)
                val_scores[feature_map] = val_scores.groupby(by, group_keys=False)[feature_map].apply(groupby_helper,min_period_per=1)
            train_scores = train_scores[train_scores['source_flag']=='train'].drop(columns=['source_flag'])
            val_scores = val_scores[val_scores['source_flag']=='val'].drop(columns=['source_flag'])
            test_scores = whole_data[whole_data['source_flag']=='test'].drop(columns=['source_flag'])
        else:
            train_scores = whole_data[whole_data['source_flag']=='train'].drop(columns=['source_flag'])
            val_scores = whole_data[whole_data['source_flag']=='val'].drop(columns=['source_flag'])
            test_scores = whole_data[whole_data['source_flag']=='test'].drop(columns=['source_flag'])
        del whole_data
        train_scores = train_scores.reset_index(drop=True)
        val_scores = val_scores.reset_index(drop=True)
        test_scores = test_scores.reset_index(drop=True)        
        



        if save2disk and pCache:
            train_scores.to_csv(savepaths['train_scores'],index=False)
            val_scores.to_csv(savepaths['val_scores'],index=False)
            test_scores.to_csv(savepaths['test_scores'],index=False)
    else:
        train_scores = pd.read_csv(savepaths['train_scores'])
        val_scores = pd.read_csv(savepaths['val_scores'])
        test_scores = pd.read_csv(savepaths['test_scores'])

    return train_scores,val_scores,test_scores

def get_val_performance_data(args, test_scores, val_scores, train_scores, gt_labels, feature_map , graph_path ,savepaths, save2disk= True, pCache=True, threshold_per=-1, verbose=True, draw_eval=True,g_extracter=None):
    saved_dir= os.path.dirname(savepaths['train_scores'])


    if os.path.exists(graph_path):
        g = pd.read_csv(graph_path)
        if g.values.shape[0]!=0:
            g = torch.Tensor(g.values)
        else:
            g = torch.empty(g.values.shape)
    else:
        g = torch.empty(0,3)
    if os.path.exists(savepaths['test_top_scores']):
        df_topk_test_scores = pd.read_csv(savepaths['test_top_scores'])
    else:
        total_topk_test_indices, total_topk_test_scores = sum_topk_scores(test_scores[feature_map].values.T, topk=1)
        col_list = []
        if 'anomaly_source' not in test_scores.columns and 'GPU' not in test_scores.columns:
            col_list = ['timestamp','instance','label']
        elif 'anomaly_source' in test_scores.columns and 'GPU' not in test_scores.columns:
            col_list = ['timestamp','instance','label','anomaly_source']
        elif 'anomaly_source' not in test_scores.columns and 'GPU' in test_scores.columns:
            col_list = ['timestamp','instance','label','GPU']
        elif 'anomaly_source' in test_scores.columns and 'GPU' in test_scores.columns:
            col_list = ['timestamp','instance','label','anomaly_source','GPU']
        df_topk_test_scores = deepcopy(test_scores[col_list])
        df_topk_test_scores['score'] = total_topk_test_scores

        df_topk_test_scores['contributor_id'] = total_topk_test_indices[0]

        df_topk_test_scores['contributor'] = [feature_map[i] for i in total_topk_test_indices[0]]
        g = g[g[:,2].sort(descending=True)[1]]
        g = g[g[:,0]!=g[:,1]]
        df_topk_test_scores['neighbors'] = [tuple(map(lambda x: feature_map[int(x)] , g[g[:,1]==des][:,0].tolist())) for des in total_topk_test_indices[0]]

    if os.path.exists(savepaths['val_top_scores']):
        df_topk_val_scores = pd.read_csv(savepaths['val_top_scores'])
        val_rec_meansquareerror = np.mean(np.square(df_topk_val_scores.score.values))
    else:
        total_topk_val_indices, total_topk_val_scores = sum_topk_scores(val_scores[feature_map].values.T, topk=1)
        val_rec_meansquareerror = np.mean(np.square(total_topk_val_scores))
        if 'anomaly_source' not in val_scores.columns:
            df_topk_val_scores = deepcopy(val_scores[['timestamp','instance','label']])
        else:
            df_topk_val_scores = deepcopy(val_scores[['timestamp','instance','label','anomaly_source']])
        df_topk_val_scores['score'] = total_topk_val_scores

        df_topk_val_scores['contributor'] = [feature_map[i] for i in total_topk_val_indices[0]]
    
    df_topk_test_scores['timestamp'] = pd.to_datetime(df_topk_test_scores['timestamp'])
    df_topk_val_scores['timestamp'] = pd.to_datetime(df_topk_val_scores['timestamp'])


    if args.eval_tol>0:
        score_column = 'score(max)'
    else:
        score_column = 'score'
        

    if save2disk :            
        if threshold_per == 'best' or  threshold_per == -1.0:
            threshold_per_list = [0.99,0.999,0.9999]
        else:
            threshold_per_list = [threshold_per]
        if len(threshold_per_list) > 1:
            print('>>>   search best f1 score')
        else:
            print('>>>   Using threshold quantile:', threshold_per_list[0])

        if args.simulate_perfect_model:
            f1,prec,rec,best_threshold_per= 1,1,1,1
            df_topk_test_scores['pred']=df_topk_test_scores['label']
        else:
            f1, pre, rec, best_threshold_per = best_f1_score(df_topk_test_scores, train_scores[feature_map], threshold_per_list, draw_flag=draw_eval, save2disk=save2disk, savepath=saved_dir+'/eval_threshold',tol=args.eval_tol,g_extracter=g_extracter,args=args)
        best_threshold = np.quantile(train_scores[feature_map].values, best_threshold_per)
        df_topk_val_scores = set_df_label(df_topk_val_scores, best_threshold, tol=args.eval_tol)


        tp = df_topk_test_scores[(df_topk_test_scores['pred']==1)& (df_topk_test_scores['label']==1)].shape[0]
        fn = df_topk_test_scores[(df_topk_test_scores['pred']==0)& (df_topk_test_scores['label']==1)].shape[0]
        fp = df_topk_test_scores[(df_topk_test_scores['pred']==1)& (df_topk_test_scores['label']==0)].shape[0]
        tn = df_topk_test_scores[(df_topk_test_scores['pred']==0)& (df_topk_test_scores['label']==0)].shape[0]
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tp_instance = df_topk_test_scores[(df_topk_test_scores['pred']==1)& (df_topk_test_scores['label']==1)]['instance'].nunique()
        fn_instance = df_topk_test_scores[(df_topk_test_scores['pred']==0)& (df_topk_test_scores['label']==1)]['instance'].nunique()
        fp_instance = df_topk_test_scores[(df_topk_test_scores['pred']==1)& (df_topk_test_scores['label']==0)]['instance'].nunique()
        tn_instance = df_topk_test_scores[(df_topk_test_scores['pred']==0)& (df_topk_test_scores['label']==0)]['instance'].nunique()

        if len(np.unique(df_topk_test_scores['label'])) != 1:
            auc_score = roc_auc_score(df_topk_test_scores['label'], df_topk_test_scores[score_column],average='macro')
        else:
            auc_score = 0
        adjust_labels, adjust_scores, adjust_weight = adjust_labels_scores_pd(df_topk_test_scores, 'label', score_column, groupby_col='instance')
        if len(np.unique(adjust_labels)) != 1:
            adjusted_auc_score = roc_auc_score(adjust_labels, adjust_scores,average='macro',sample_weight=adjust_weight)
        else:
            adjusted_auc_score = 0

        if 'anomaly_source' in df_topk_test_scores.columns:
            df_topk_test_scores_temp = merge_label_with_cause(df_topk_test_scores,g_extracter=g_extracter)

            f1_graph = f1_score(df_topk_test_scores_temp['label'], df_topk_test_scores_temp['pred'],average='macro')
            tp_graph = df_topk_test_scores_temp[(df_topk_test_scores_temp['pred']==1)& (df_topk_test_scores_temp['label']==1)].shape[0]
            fn_graph = df_topk_test_scores_temp[(df_topk_test_scores_temp['pred']==0)& (df_topk_test_scores_temp['label']==1)].shape[0]
            fp_graph = df_topk_test_scores_temp[(df_topk_test_scores_temp['pred']==1)& (df_topk_test_scores_temp['label']==0)].shape[0]
            tn_graph = df_topk_test_scores_temp[(df_topk_test_scores_temp['pred']==0)& (df_topk_test_scores_temp['label']==0)].shape[0]
        else:
            f1_graph = 0
            tp_graph = 0
            fn_graph = 0
            fp_graph = 0
            tn_graph = 0
            
        if hasattr(args,'tol_min'):
            delays = cal_delay(df_topk_test_scores_temp, args.tol_min)
            num_delay = delays[delays!=0].shape[0]
            if num_delay == 0:
                average_delay = -1
            else:
                average_delay = delays[delays!=0].mean()

        if 'GPU' in df_topk_test_scores.columns:
            df_topk_test_scores = find_anomaly_gpu(df_topk_test_scores)


        root_cause_identification_score = -1
        root_cause_identification_all_cause_match_score = -1
        root_cause_identification_all_cause_neighbors_match_score = -1
        root_cause_identification_neighbors_match_score = -1
        avg_num_causes = -1
        avg_num_all_causes = -1

        if False:
            if ((df_topk_test_scores['label']==1) & (df_topk_test_scores['pred']==1)).sum()!=0:
                df_topk_test_scores_temp = merge_label_with_cause(df_topk_test_scores,g_extracter=g_extracter)
                if df_topk_test_scores['anomaly_source'].dtype=='object':
                    fillna_value = 'None'
                elif df_topk_test_scores['anomaly_source'].dtype=='float' or df_topk_test_scores['anomaly_source'].dtype=='int':
                    fillna_value = -1
                root_cause_identification_score = precision_score(
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['anomaly_source'].fillna(fillna_value,inplace=False),
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['cause_pred'].fillna(fillna_value,inplace=False),
                    average="micro",
                )
                df_topk_test_scores_temp = get_matched_pred(df_topk_test_scores_temp, type='with_neighbors')
                root_cause_identification_neighbors_match_score = precision_score(
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['anomaly_source'].fillna(fillna_value,inplace=False),
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['matched_pred'].fillna(fillna_value,inplace=False),
                    average="micro",
                )

                df_topk_test_scores_temp = get_matched_pred(df_topk_test_scores_temp, type='all_cause_pred')
                root_cause_identification_all_cause_match_score = precision_score(
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['anomaly_source'].fillna(fillna_value,inplace=False),
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['matched_pred'].fillna(fillna_value,inplace=False),
                    average="micro",
                )
                df_topk_test_scores_temp = get_matched_pred(df_topk_test_scores_temp, type='all_cause_pred_with_neighbors')
                root_cause_identification_all_cause_neighbors_match_score = precision_score(
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['anomaly_source'].fillna(fillna_value,inplace=False),
                    df_topk_test_scores_temp[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1)]['matched_pred'].fillna(fillna_value,inplace=False),
                    average="micro",
                )
                avg_num_causes= df_topk_test_scores_temp.loc[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1),'all_cause_pred'].apply(len).mean()
                avg_num_all_causes= df_topk_test_scores_temp.loc[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1),'all_cause_pred'].apply(len).mean()+ df_topk_test_scores_temp.loc[(df_topk_test_scores_temp['label']==1) & (df_topk_test_scores_temp['pred']==1),'all_cause_pred_neighbors'].apply(len).mean()



        if 'anomaly_source' in df_topk_test_scores.columns:
            if not hasattr(args,'tol_min'):
                df_best_eval_best = pd.DataFrame(columns=['best_threshold_per','f1','precision','recall','auc_score(adjusted)','auc_score(native)','root_cause_identification_score','root_cause_identification_all_cause_match_score','root_cause_identification_all_cause_neighbors_match_score','root_cause_identification_neighbors_match_score','avg_num_causes','avg_num_causes(graph)','tp','fn','fp','tn','tpr','fpr','tp_instance','fn_instance','fp_instance','tn_instance','tp_graph','fn_graph','fp_graph','tn_graph','f1_graph'])
                df_best_eval_best.loc[0] = [best_threshold_per,f1,pre,rec,adjusted_auc_score,auc_score,root_cause_identification_score,root_cause_identification_all_cause_match_score,root_cause_identification_all_cause_neighbors_match_score,root_cause_identification_neighbors_match_score,avg_num_causes,avg_num_all_causes,tp,fn,fp,tn,tpr,fpr,tp_instance,fn_instance,fp_instance,tn_instance,tp_graph,fn_graph,fp_graph,tn_graph,f1_graph]
            else:
                df_best_eval_best = pd.DataFrame(columns=['best_threshold_per','f1','precision','recall','auc_score(adjusted)','auc_score(native)','root_cause_identification_score','root_cause_identification_all_cause_match_score','root_cause_identification_all_cause_neighbors_match_score','root_cause_identification_neighbors_match_score','avg_num_causes','avg_num_causes(graph)','tp','fn','fp','tn','tpr','fpr','tp_instance','fn_instance','fp_instance','tn_instance','tp_graph','fn_graph','fp_graph','tn_graph','f1_graph','delay','num_delay'])
                df_best_eval_best.loc[0] = [best_threshold_per,f1,pre,rec,adjusted_auc_score,auc_score,root_cause_identification_score,root_cause_identification_all_cause_match_score,root_cause_identification_all_cause_neighbors_match_score,root_cause_identification_neighbors_match_score,avg_num_causes,avg_num_all_causes,tp,fn,fp,tn,tpr,fpr,tp_instance,fn_instance,fp_instance,tn_instance,tp_graph,fn_graph,fp_graph,tn_graph,f1_graph,average_delay,num_delay]
        else:
            df_best_eval_best = pd.DataFrame(columns=['best_threshold_per','f1','precision','recall','auc_score(adjusted)','auc_score(native)'])
            df_best_eval_best.loc[0] = [best_threshold_per,f1,pre,rec,adjusted_auc_score,auc_score]


        df_best_eval_best = df_best_eval_best.T
        if save2disk and pCache:
            df_topk_test_scores.to_csv(savepaths['test_top_scores'],index=False)
            df_topk_val_scores.to_csv(savepaths['val_top_scores'],index=False)
        if save2disk:
            df_best_eval_best.to_csv(savepaths['best_eval_result'],index=True)

    else:
        best_eval = pd.read_csv(savepaths['best_eval_result'],index_col=0).T
        best_threshold_per = best_eval['best_threshold_per'].values[0]
        pre = best_eval['precision'].values[0]
        rec = best_eval['recall'].values[0]
        f1 = best_eval['f1'].values[0]
        auc_score = best_eval['auc_score(native)'].values[0]
        adjusted_auc_score = best_eval['auc_score(adjusted)'].values[0]
        if 'anomaly_source' in df_topk_test_scores.columns and df_topk_test_scores['anomaly_source'].dtype=='object':
            root_cause_identification_score = best_eval['root_cause_identification_score'].values[0]
        
        if threshold_per == 'best' or  threshold_per== -1:
            print('re-search best f1 score')
            step = 0.0001
            threshold_per_list = np.arange(0.99,1+step,step) 
            f1, pre, rec, best_threshold_per = best_f1_score(df_topk_test_scores, train_scores[feature_map], threshold_per_list, draw_flag=draw_eval, save2disk=save2disk, savepath=saved_dir+'/eval_threshold',tol=args.args)
            best_eval['best_threshold_per'] = best_threshold_per
            best_eval['f1'] = f1
            best_eval = best_eval.T
            best_eval.to_csv(savepaths['best_eval_result'],index=True)
            
    if draw_eval and save2disk:
        test_contributor_id = df_topk_test_scores.contributor_id.values
        
    if verbose:
        print(f'F1 score: {f1}')
        print(f'precision: {pre}')
        print(f'recall: {rec}')
        print(f'adjusted auroc: {adjusted_auc_score}\n')
        print(f'auroc: {auc_score}')
        if 'anomaly_source' in df_topk_test_scores.columns and df_topk_test_scores['anomaly_source'].dtype=='object':
            print(f'root cause identification score: {root_cause_identification_score}\n')

    return f1, pre, rec, auc_score, adjusted_auc_score, val_rec_meansquareerror

def get_instance_level_df(df,label_col='label',pred_col='pred',score_col='score',cause_gt_col='anomaly_source',cause_pred_col='contributor',groupby_col='instance',save2disk=True,savepath='',skip_causes=[]):
    def helper(group):
        label = group[label_col].max()
        pred = group[pred_col].max()

        pred_with_tol = 0 if group['pred'].sum() <= 3 else 1
        score = group[score_col].max()
        if cause_gt_col in group.columns:
            if label == 0:
                cause_gt = None
            else:
                if group[cause_gt_col].dropna().unique().shape[0]== 1:
                    cause_gt = group[cause_gt_col].dropna().iloc[0]
                elif group[cause_gt_col].dropna().unique().shape[0] > 1:
                    cause_gt = group[group[label_col]==1][cause_gt_col].value_counts().index[0]
                elif group[cause_gt_col].dropna().unique().shape[0] == 0:
                    raise ValueError("label and anomaly_source are inconsistent!")

            if pred_with_tol ==0 or label==0:
                cause_pred = None
                cause_pred_neighbors = None
                all_cause_pred = None
                all_cause_pred_neighbors = None
            else:
                if group[(group[pred_col]==1) & (group[label_col]==1)][cause_pred_col].unique().shape[0] == 0:
                    cause_pred = None
                    cause_pred_neighbors = None
                    all_cause_pred = None
                    all_cause_pred_neighbors = None
                elif group[(group[pred_col]==1) & (group[label_col]==1)][cause_pred_col].unique().shape[0] == 1:
                    cause_pred = group[(group[pred_col]==1) & (group[label_col]==1)][cause_pred_col].iloc[0]
                    cause_pred_neighbors = group[(group[pred_col]==1) & (group[label_col]==1)]['neighbors'].iloc[0]
                    all_cause_pred = cause_pred
                    all_cause_pred_neighbors = group[(group[pred_col]==1) & (group[label_col]==1)]['neighbors'].unique()
                    all_cause_pred_neighbors = tuple(set([elem for sublist in all_cause_pred_neighbors for elem in sublist]))
                else:
                    if skip_causes != []:
                        group = group.sort_values(by=score_col,ascending=False)
                        cause_pred = group[(group[pred_col]==1) & (group[label_col]==1) & (~group[cause_pred_col].isin(skip_causes))][cause_pred_col].value_counts(sort=False).index[0]
                        cause_pred_neighbors = group[(group[pred_col]==1) & (group[label_col]==1) & (~group[cause_pred_col].isin(skip_causes))]['neighbors'].value_counts(sort=False).index[0]
                        all_cause_pred = group[(group[pred_col]==1) & (group[label_col]==1) & (~group[cause_pred_col].isin(skip_causes))][cause_pred_col].value_counts(sort=False).index.tolist()
                        all_cause_pred_neighbors = group[(group[pred_col]==1) & (group[label_col]==1) & (~group[cause_pred_col].isin(skip_causes))]['neighbors'].unique()
                        all_cause_pred_neighbors = tuple(set([elem for sublist in all_cause_pred_neighbors for elem in sublist]))
                    else:
                        group = group.sort_values(by=score_col,ascending=False)
                        cause_pred = group[(group[pred_col]==1) & (group[label_col]==1)][cause_pred_col].value_counts(sort=False).index[0]
                        cause_pred_neighbors = group[(group[pred_col]==1) & (group[label_col]==1)]['neighbors'].value_counts(sort=False).index[0]
                        all_cause_pred = group[(group[pred_col]==1) & (group[label_col]==1)][cause_pred_col].value_counts(sort=False).index.tolist()
                        all_cause_pred_neighbors = group[(group[pred_col]==1) & (group[label_col]==1)]['neighbors'].unique()
                        all_cause_pred_neighbors = tuple(set([elem for sublist in all_cause_pred_neighbors for elem in sublist]))
            
            
            return pd.DataFrame([[label,pred_with_tol,score,cause_gt,cause_pred,all_cause_pred,all_cause_pred_neighbors,cause_pred_neighbors]],columns=['label','pred',score_col,'cause_gt','cause_pred','all_cause_pred','all_cause_pred_neighbors','cause_pred_neighbors'])
        else:
            return pd.DataFrame([[label,pred_with_tol,score]], columns=['label','pred',score_col])
        
    gt_nodeslabel = df.groupby(groupby_col, group_keys=False).apply(helper)
    if save2disk:
        gt_nodeslabel.to_csv(savepath,index=True)
    return gt_nodeslabel

def merge_label_with_cause(df, native_pred=False, contributor_col='contributor', neightbors_col='neighbors', skip_causes=[],g_extracter=None):
    df_copy = deepcopy(df)
    def adjust_pred_cause(group):
        instance = group.name
        group['sim'] = -1
        if not group['timestamp'].is_monotonic_increasing:
            raise ValueError("index_column must be monotonic")
        group['cause_pred'] = None
        group['all_cause_pred'] = None
        group['all_cause_pred_neighbors'] = None
        group['cause_pred_neighbors'] = None
        if native_pred:
            pred_col= 'pred_native'
        else:
            pred_col = 'pred'
        group.loc[group[pred_col]==1,'cause_pred'] = group.loc[group[pred_col]==1,contributor_col] 
        group.loc[group[pred_col]==1,'cause_pred_neighbors'] = group.loc[group[pred_col]==1,neightbors_col]
        if group['pred'].unique().shape[0] > 1:
            group['block'] = (group['label'].diff(1) != 0) & (group['label'] == 1)
            group['block_num'] = group['block'].cumsum()
            masks = [group['block_num'] == i for i in range(group['block_num'].max()+1)]
            for mask in masks:
                mask1 = mask & (group['pred'] == 1) & (group['label'] == 1)
                mask2 = mask & (group['pred'] == 1) & (group['label'] == 0)
                nunique_cause_mask1= group.loc[mask1, contributor_col].unique().shape[0]
                nunique_cause_mask2= group.loc[mask2, contributor_col].unique().shape[0]
                if  nunique_cause_mask1 == 1:
                    if g_extracter is not None:
                        cur_graph_mean,_ = g_extracter.profile_graph_4hours(group[mask1], time_win = 1, history_shift=0,instance = instance)
                        if len(cur_graph_mean.shape) != 2: continue
                        prev_graph_mean,_ = g_extracter.profile_graph_4hours(group[mask1], time_win = 1, history_shift=1,instance = instance)
                        if len(prev_graph_mean.shape) != 2: continue
                        prev_g = pd.DataFrame(convert_adj2edges(prev_graph_mean),columns=['source','destination','value'])
                        prev_g['source'] = prev_g['source'].apply(lambda x: int(x))
                        prev_g['destination'] = prev_g['destination'].apply(lambda x: int(x))
                        g = pd.DataFrame(convert_adj2edges(cur_graph_mean),columns=['source','destination','value'])
                        g['source'] = g['source'].apply(lambda x: int(x))
                        g['destination'] = g['destination'].apply(lambda x: int(x))
                        contributor_maxscore_id = group.loc[mask1, 'contributor_id'].iloc[0]

                        similarity = get_similarity(prev_g,g,distance='common')
                        group['sim'] = similarity

                        if (group.loc[mask1,'label']==1).all():
                            g_extracter.tp_sim_list.append([instance,similarity])
                        elif (group.loc[mask1,'label']==0).all():
                            g_extracter.fp_sim_list.append([instance,similarity])


                    group.loc[mask1, 'cause_pred'] = group.loc[mask1, contributor_col].iloc[0]
                    group.loc[mask1, 'all_cause_pred'] = pd.Series([tuple([group.loc[mask1, contributor_col].iloc[0]])]*mask1.sum(), index=group.loc[mask1].index)
                    if g_extracter is not None:
                        all_cause_pred_neighbors = set(g[g['destination']==contributor_maxscore_id][:5]['source'].values)
                        all_cause_pred_neighbors = set(map(g_extracter.id2feature,all_cause_pred_neighbors))
                    else:
                        all_cause_pred_neighbors = group.loc[mask1, neightbors_col].iloc[0]
                    group.loc[mask1, 'all_cause_pred_neighbors'] = pd.Series([tuple(set([elem for elem in all_cause_pred_neighbors]))]*mask1.sum(), index=group.loc[mask1].index)
                    group.loc[mask1, 'cause_pred_neighbors'] = pd.Series([tuple(set([elem for elem in all_cause_pred_neighbors]))]*mask1.sum(), index=group.loc[mask1].index)
                elif nunique_cause_mask1 > 1:
                    if native_pred:
                        mask_temp = mask1 & (group['pred_native'] == 1)
                    else:
                        mask_temp = mask1
                    if skip_causes != []:
                        mask_temp = mask_temp & ~group[contributor_col].isin(skip_causes)
                    idx_maxscore = group.loc[mask1,'score'].idxmax()
                    contributor_maxscore = group.loc[idx_maxscore][contributor_col]
                    group.loc[mask1, 'cause_pred'] = contributor_maxscore

                    group_sumscore = group.loc[mask1].groupby('contributor').sum('score')
                    contributor_other = group_sumscore['score'].idxmax()
                    all_contributors = set([contributor_maxscore,contributor_other])

                    if g_extracter is not None:
                        cur_graph_mean,_ = g_extracter.profile_graph_4hours(group[mask1], time_win = 1, history_shift=0,instance = instance)
                        if len(cur_graph_mean.shape) != 2: continue
                        prev_graph_mean,_ = g_extracter.profile_graph_4hours(group[mask1], time_win = 1, history_shift=1,instance = instance)
                        if len(prev_graph_mean.shape) != 2: continue
                        prev_g = pd.DataFrame(convert_adj2edges(prev_graph_mean),columns=['source','destination','value'])
                        prev_g['source'] = prev_g['source'].apply(lambda x: int(x))
                        prev_g['destination'] = prev_g['destination'].apply(lambda x: int(x))
                        g = pd.DataFrame(convert_adj2edges(cur_graph_mean),columns=['source','destination','value'])
                        g['source'] = g['source'].apply(lambda x: int(x))
                        g['destination'] = g['destination'].apply(lambda x: int(x))

                        contributor_maxscore_id = group.loc[idx_maxscore]['contributor_id']
                        if len(all_contributors)>1:
                            contributor_other_id = group[(group[contributor_col]==contributor_other) & mask1].iat[0, group.columns.get_loc('contributor_id')]

                        
                        similarity = get_similarity(prev_g,g,distance='common')
                        group['sim'] = similarity
                        if (group.loc[mask1,'label']==1).all():
                            g_extracter.tp_sim_list.append([instance,similarity])
                        elif (group.loc[mask1,'label']==0).all():
                            g_extracter.fp_sim_list.append([instance,similarity])
                    group.loc[mask1, 'all_cause_pred'] = pd.Series([tuple(all_contributors)]*mask1.sum(), index=group.loc[mask1].index)
                    
                    if g_extracter is not None:
                        cause_pred_neighbors = set(g[g['destination']==contributor_maxscore_id][:5]['source'].values)
                        cause_pred_neighbors = set(map(g_extracter.id2feature,cause_pred_neighbors))
                    else:
                        cause_pred_neighbors = set(group.loc[idx_maxscore, neightbors_col])
                    group.loc[mask1, 'cause_pred_neighbors'] = pd.Series([tuple(cause_pred_neighbors)]*mask1.sum(), index=group.loc[mask1].index)
                    
                    if len(all_contributors) == 1:
                        all_cause_pred_neighbors = cause_pred_neighbors
                    else:
                        if g_extracter is not None:
                            all_cause_pred_neighbors = set(g[g['destination']==contributor_other_id][:5]['source'].values)
                            all_cause_pred_neighbors = set(map(g_extracter.id2feature,all_cause_pred_neighbors))
                        else:
                            other_cause_pred_neighbors = set(group[(group[contributor_col]==contributor_other) & mask1].iat[0, group.columns.get_loc(neightbors_col)])
                            all_cause_pred_neighbors = other_cause_pred_neighbors.union(cause_pred_neighbors)
                    
                    group.loc[mask1, 'all_cause_pred_neighbors'] = pd.Series([tuple(all_cause_pred_neighbors)]*mask1.sum(), index=group.loc[mask1].index)

                if nunique_cause_mask2 >= 1:
                    if g_extracter is not None:
                        for i in range(group[mask2].shape[0]):
                            first_selected_index  = group[mask2].index[i]
                            group_row = group.loc[[first_selected_index]]
                            cause_id = group_row['contributor_id'].iloc[0]
                            cur_graph_mean,_ = g_extracter.profile_graph_4hours(group_row, time_win = 1, history_shift=0,instance = instance)
                            if len(cur_graph_mean.shape) != 2: continue
                            prev_graph_mean,_ = g_extracter.profile_graph_4hours(group_row, time_win = 1, history_shift=1,instance = instance)
                            if len(prev_graph_mean.shape) != 2: continue
                            prev_g = pd.DataFrame(convert_adj2edges(prev_graph_mean),columns=['source','destination','value'])
                            prev_g['source'] = prev_g['source'].apply(lambda x: int(x))
                            prev_g['destination'] = prev_g['destination'].apply(lambda x: int(x))
                            g = pd.DataFrame(convert_adj2edges(cur_graph_mean),columns=['source','destination','value'])
                            g['source'] = g['source'].apply(lambda x: int(x))
                            g['destination'] = g['destination'].apply(lambda x: int(x))
                            similarity = get_similarity(prev_g,g,distance='common')
                            group['sim'] = similarity
                            if (group_row['label']==1).all():
                                g_extracter.tp_sim_list.append([instance,similarity])
                            elif (group_row['label']==0).all():
                                g_extracter.fp_sim_list.append([instance,similarity])

            group.drop(columns=['block','block_num'],inplace=True)

        group = group.replace({np.nan: None})
        return group
    df_copy = df_copy.groupby('instance', group_keys=False).apply(adjust_pred_cause)
    if g_extracter is not None:
        g_extracter.tp_sim_list = pd.DataFrame(g_extracter.tp_sim_list,columns=['farmnode','value'])
        g_extracter.fp_sim_list = pd.DataFrame(g_extracter.fp_sim_list,columns=['farmnode','value'])
        tp_sim_list = df_copy[(df_copy['pred'] == 1) & (df_copy['label'] == 1)]['sim']
        best_graph_thr = tp_sim_list.min()
        df_copy.loc[(df_copy['pred'] == 1) & (df_copy['label'] == 0),'pred'] = df_copy.loc[(df_copy['pred'] == 1) & (df_copy['label'] == 0)]['sim'].apply(lambda x: 0 if x < best_graph_thr else 1)
    return df_copy

def get_matched_pred(df, type='all_cause_pred',cause_gt_col='anomaly_source'):
    
    def match_all_cause_pred(group):
        cause_gt=group[cause_gt_col]
        if cause_gt == None or group['all_cause_pred'] == None:
            return group['cause_pred']
        if cause_gt in group['all_cause_pred']:
            return cause_gt
        else:
            return group['cause_pred']
        
    def match_all_cause_pred_with_neighbors(group):
        cause_gt=group[cause_gt_col]
        if cause_gt == None or group['all_cause_pred'] == None:
            return group['cause_pred']
        if cause_gt in group['all_cause_pred']:
            return cause_gt
        elif cause_gt!=None and cause_gt in group['all_cause_pred_neighbors']:
            return cause_gt
        else:
            return group['cause_pred']
        
    def match_with_neighbors(group):
        cause_gt=group[cause_gt_col]
        if cause_gt == None or group['cause_pred_neighbors'] == None:
            return group['cause_pred']
        elif cause_gt!=None and cause_gt in group['cause_pred_neighbors']:
            return cause_gt
        else:
            return group['cause_pred']

    df['matched_pred'] = df['cause_pred']
    if type == 'all_cause_pred':
        mask = df['pred']==1
        df.loc[mask,'matched_pred'] = df.loc[mask].apply(match_all_cause_pred, axis=1)
    elif type == 'all_cause_pred_with_neighbors':
        mask = df['pred']==1
        df.loc[mask,'matched_pred'] = df.loc[mask].apply(match_all_cause_pred_with_neighbors, axis=1)
    elif type == 'with_neighbors':
        mask = df['pred']==1
        df.loc[mask,'matched_pred'] = df.loc[mask].apply(match_with_neighbors, axis=1)
    return df

def find_anomaly_gpu(df):
    contributor_gpu_map = {
        "p0_gpu0_power": 0,
        "p0_gpu1_power": 1,
        "p0_gpu2_power": 2,
        "p1_gpu0_power": 3,
        "p1_gpu1_power": 4,
        "p1_gpu2_power": 5,
        "gpu0_core_temp": 0,
        "gpu0_mem_temp": 0,
        "gpu1_core_temp": 1,
        "gpu1_mem_temp": 1,
        "gpu2_core_temp": 2,
        "gpu2_mem_temp": 2,
        "gpu3_core_temp": 3,
        "gpu3_mem_temp": 3,
        "gpu4_core_temp": 4,
        "gpu4_mem_temp": 4,
        "gpu5_core_temp": 5,
        "gpu5_mem_temp":5,
    }
    df['anomaly_source'] = df['GPU'].astype(int)
    df['contributor'] = df['contributor'].map(lambda x: contributor_gpu_map[x] if x in contributor_gpu_map.keys() else -1)
    df['neighbors'] = df['neighbors'].map(lambda x: tuple([contributor_gpu_map[elem] for elem in x if elem in contributor_gpu_map.keys()]))
    return df

def find_anomaly_slot(df):
    contributor_gpu_map = {
        "p0_gpu0_power": 0,
        "p0_gpu1_power": 0,
        "p0_gpu2_power": 0,
        "p1_gpu0_power": 1,
        "p1_gpu1_power": 1,
        "p1_gpu2_power": 1,
        "gpu0_core_temp": 0,
        "gpu0_mem_temp": 0,
        "gpu1_core_temp": 0,
        "gpu1_mem_temp": 0,
        "gpu2_core_temp": 0,
        "gpu2_mem_temp": 0,
        "gpu3_core_temp": 1,
        "gpu3_mem_temp": 1,
        "gpu4_core_temp": 1,
        "gpu4_mem_temp": 1,
        "gpu5_core_temp": 1,
        "gpu5_mem_temp":1,
    }
    df['anomaly_source'] = df['GPU'].astype(int)
    df['anomaly_source'] = df['anomaly_source'].map(lambda x: 0 if x in [0,1,2] else 1)
    df['contributor'] = df['contributor'].map(lambda x: contributor_gpu_map[x] if x in contributor_gpu_map.keys() else -1)
    df['neighbors'] = df['neighbors'].map(lambda x: tuple([contributor_gpu_map[elem] for elem in x if elem in contributor_gpu_map.keys()]))
    return df

def find_anomaly_device(df):
    contributor_gpu_map = {
        "p0_gpu0_power": 1,
        "p0_gpu1_power": 1,
        "p0_gpu2_power": 1,
        "p1_gpu0_power": 1,
        "p1_gpu1_power": 1,
        "p1_gpu2_power": 1,
        "gpu0_core_temp": 1,
        "gpu0_mem_temp": 1,
        "gpu1_core_temp": 1,
        "gpu1_mem_temp": 1,
        "gpu2_core_temp": 1,
        "gpu2_mem_temp": 1,
        "gpu3_core_temp": 1,
        "gpu3_mem_temp": 1,
        "gpu4_core_temp": 1,
        "gpu4_mem_temp": 1,
        "gpu5_core_temp": 1,
        "gpu5_mem_temp":1,
    }
    df['anomaly_source'] = df['GPU'].astype(int)
    df['anomaly_source'] = df['anomaly_source'].map(lambda x: 1 if x>=0 else 0)
    df['contributor'] = df['contributor'].map(lambda x: contributor_gpu_map[x] if x in contributor_gpu_map.keys() else -1)
    df['neighbors'] = df['neighbors'].map(lambda x: tuple([contributor_gpu_map[elem] for elem in x if elem in contributor_gpu_map.keys()]))
    return df


def best_f1_score(test_scores, train_scores, threshold_per_list= np.arange(0.9,1,0.001),draw_flag=False, save2disk=False,savepath="",tol=0,g_extracter=None,args=None):
    f1_list = []
    prec_list = []
    recall_list = []
    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    tpr_list = []
    fpr_list = []
    tp_graph_list = []
    fn_graph_list = []
    fp_graph_list = []
    tn_graph_list = []
    f1_graph_list = []
    if hasattr(args,'tol_min'):
        delay_list = []
        num_delay_list = []
    best_f1 = 0
    best_index = 0
    best_pred = None
    best_pred_native = None
    best_pred_direct = None
    if len(threshold_per_list) == 1:
        draw_flag = False
    test_scores_copy = deepcopy(test_scores)
    if tol > 0:
        score_column = 'score(max)'
        win_size = 10
        assert test_scores_copy['timestamp'].is_monotonic_increasing
        test_scores_copy = test_scores_copy.groupby('instance', group_keys=False).apply(rolling_max_helper, ['score'], win_size)
        test_scores[score_column] = test_scores_copy[score_column].values
    else:
        score_column = 'score'
    if draw_flag:
        h_fig = make_subplots(rows=1, cols=1)
        labels_name = ['gt_normal','gt_anomaly']
        pred_labels_name = ['pred_normal','pred_anomaly']
        selected_threshold_per_list = np.arange(0.99,1+0.001,0.001)


    for idx, threshold_per in enumerate(threshold_per_list):
        threshold = np.quantile(train_scores.values, threshold_per)
        test_pred_labels = np.zeros(len(test_scores_copy[score_column])).astype(int)
        test_pred_labels[test_scores_copy[score_column] > threshold] = 1
        test_scores_copy['pred'] = test_pred_labels
        test_scores_copy = adjust_predicts_pd(test_scores_copy, 'label', 'pred', groupby_col='instance',tol=tol)

        prec = precision_score(test_scores_copy['label'], test_scores_copy['pred'],average='macro')
        rec = recall_score(test_scores_copy['label'], test_scores_copy['pred'],average='macro')
        f1 = f1_score(test_scores_copy['label'], test_scores_copy['pred'],average='macro')
        f1_list.append(f1)
        prec_list.append(prec)
        recall_list.append(rec)

        tp= test_scores_copy[(test_scores_copy['pred']==1)& (test_scores_copy['label']==1)].shape[0]
        fn= test_scores_copy[(test_scores_copy['pred']==0)& (test_scores_copy['label']==1)].shape[0]
        fp= test_scores_copy[(test_scores_copy['pred']==1)& (test_scores_copy['label']==0)].shape[0]
        tn= test_scores_copy[(test_scores_copy['pred']==0)& (test_scores_copy['label']==0)].shape[0]

        tp_list.append(tp)
        fn_list.append(fn)
        fp_list.append(fp)
        tn_list.append(tn)
        tpr_list.append(tp/(tp+fn))
        fpr_list.append(fp/(fp+tn))

        test_scores_copy_temp = merge_label_with_cause(test_scores_copy,g_extracter=g_extracter)
        f1_graph = f1_score(test_scores_copy_temp['label'], test_scores_copy_temp['pred'],average='macro')
        tp_graph = test_scores_copy_temp[(test_scores_copy_temp['pred']==1)& (test_scores_copy_temp['label']==1)].shape[0]
        fn_graph = test_scores_copy_temp[(test_scores_copy_temp['pred']==0)& (test_scores_copy_temp['label']==1)].shape[0]
        fp_graph = test_scores_copy_temp[(test_scores_copy_temp['pred']==1)& (test_scores_copy_temp['label']==0)].shape[0]
        tn_graph = test_scores_copy_temp[(test_scores_copy_temp['pred']==0)& (test_scores_copy_temp['label']==0)].shape[0]

        f1_graph_list.append(f1_graph)
        tp_graph_list.append(tp_graph)
        fn_graph_list.append(fn_graph)
        fp_graph_list.append(fp_graph)
        tn_graph_list.append(tn_graph)

        if hasattr(args,'tol_min'):
            delays = cal_delay(test_scores_copy_temp, args.tol_min)
            num_delay = delays[delays!=0].shape[0]
            if num_delay == 0:
                average_delay = -1
            else:
                average_delay = delays[delays!=0].mean()
            delay_list.append(average_delay)
            num_delay_list.append(num_delay)

        if f1 > best_f1:
            best_f1 = f1
            best_index = idx
            best_pred = deepcopy(test_scores_copy['pred'].values)
            if 'pred_direct' in test_scores_copy.columns:
                best_pred_direct = deepcopy(test_scores_copy['pred_direct'].values)
            best_pred_native = test_pred_labels
        if draw_flag and round(threshold_per, 4)  in selected_threshold_per_list:
            z= []
            gt_labels_mask = test_scores_copy['label']==0
            pred_anomaly_num = np.sum(test_scores_copy['pred'][gt_labels_mask])
            z.append([np.sum(gt_labels_mask)-pred_anomaly_num,pred_anomaly_num])
            gt_labels_mask = test_scores_copy['label']==1
            pred_anomaly_num = np.sum(test_scores_copy['pred'][gt_labels_mask])
            z.append([np.sum(gt_labels_mask)-pred_anomaly_num,pred_anomaly_num])
            z = np.array(z)
            z = (z / np.sum(z,axis=1,keepdims=True))
            trace = go.Heatmap(
                visible=False, 
                z=z, 
                zmin=0,
                zmax=1,
                x=pred_labels_name, 
                y=labels_name,
                colorscale='Oranges',
            )
            h_fig.update_yaxes(autorange='reversed')
            h_fig.add_trace(trace)

    test_scores['pred']= best_pred
    test_scores['pred_native']= best_pred_native
    if 'pred_direct' in test_scores_copy.columns:
        test_scores['pred_direct']= best_pred_direct
    f1_list = np.array(f1_list)
    prec_list = np.array(prec_list)
    recall_list = np.array(recall_list)
    tp_list = np.array(tp_list)
    fp_list = np.array(fp_list)
    tn_list = np.array(tn_list)
    fn_list = np.array(fn_list)
    tpr_list = np.array(tpr_list)
    fpr_list = np.array(fpr_list)
    f1_graph_list = np.array(f1_graph_list)
    tp_graph_list = np.array(tp_graph_list)
    fn_graph_list = np.array(fn_graph_list)
    fp_graph_list = np.array(fp_graph_list)
    tn_graph_list = np.array(tn_graph_list)
    
    threshold_per_list = np.array(threshold_per_list)
    if save2disk:
        save_df = pd.DataFrame(columns=['threshold_per','f1','precision','recall','tp','fp','tn','fn','tpr','fpr','tp_graph','fn_graph','fp_graph','tn_graph','f1_graph'])
        save_df['threshold_per'] = threshold_per_list
        save_df['f1'] = f1_list
        save_df['precision'] = prec_list
        save_df['recall'] = recall_list
        save_df['tp'] = tp_list
        save_df['fp'] = fp_list
        save_df['tn'] = tn_list
        save_df['fn'] = fn_list
        save_df['tpr'] = tpr_list
        save_df['fpr'] = fpr_list
        save_df['tp_graph'] = tp_graph_list
        save_df['fn_graph'] = fn_graph_list
        save_df['fp_graph'] = fp_graph_list
        save_df['tn_graph'] = tn_graph_list
        save_df['f1_graph'] = f1_graph_list
        if hasattr(args,'tol_min'):
            save_df['delay'] = np.array(delay_list)
            save_df['num_delay'] = np.array(num_delay_list)

        save_df.to_csv(savepath+'.csv',index=False)
    if draw_flag:
        h_fig.data[0].visible = True
        steps = []
        for i, threshold in enumerate(selected_threshold_per_list):
            step = dict(
                method='update',
                args=[{'visible': [s == i for s in range(len(selected_threshold_per_list))]}],
                label='Threshold: ' + str(threshold)
            )
            steps.append(step)
        sliders = [dict(
            active=0,
            currentvalue={'prefix': 'Threshold: '},
            steps=steps
        )]
        h_fig.update_layout(sliders=sliders)
        h_fig.write_html(savepath+'_heatmap.html')

        fig = go.Figure()
        fig.add_scattergl(x=threshold_per_list,y=f1_list,name= 'f1')
        fig.add_scattergl(x=threshold_per_list,y=prec_list,name= 'precision')
        fig.add_scattergl(x=threshold_per_list,y=recall_list,name= 'recall')
        fig.update_layout(
            title= 'f1, precision, recall vs threshold',
            xaxis_title="threshold",
            yaxis_title="f1, precision, recall",
        )
        fig.write_html(savepath+'.html')
    return f1_list[best_index], prec_list[best_index], recall_list[best_index], threshold_per_list[best_index]

def set_df_label(df, threshold, score_column='score', tol=0):
    if tol > 0:
        score_column = 'score(max)'
        if score_column not in df.columns:
            win_size = 10
            assert df['timestamp'].is_monotonic_increasing
            df = df.groupby('instance', group_keys=False).apply(rolling_max_helper, ['score'], win_size)

    train_pred_labels = np.zeros(len(df))
    train_pred_labels[df[score_column] > threshold] = 1
    df['pred'] = train_pred_labels.astype(int)
    df['pred_native'] = train_pred_labels.astype(int)
    df = adjust_predicts_pd(df, 'label', 'pred', groupby_col='instance',tol=tol)
    return df

def adjust_labels_scores_pd(df, label_col, score_column, groupby_col=None, index_column='timestamp'):
    copy_df = deepcopy(df[[index_column,label_col,score_column,groupby_col]])
    def helper(df):
        if not df[index_column].is_monotonic_increasing:
            raise ValueError("index_column must be monotonic")
        new_labels = [df[label_col].iloc[0]]
        new_scores = [df[score_column].iloc[0]]
        weights = [1]
        max_score = df[score_column].iloc[0]

        labels = df[label_col].values
        scores = df[score_column].values
        for i in range(1, len(df)):
            if labels[i] ==1 and labels[i-1] == 1:
                new_scores[-1]=max(new_scores[-1], scores[i])
                weights[-1] += 1
            else:
                new_labels.append(labels[i])
                new_scores.append(scores[i])
                weights.append(1)
        return pd.Series([new_labels, new_scores,weights], index=[label_col, score_column,'weights'])
    result = copy_df.groupby(groupby_col, group_keys=False).apply(helper)
    result = result.apply(pd.Series.explode).reset_index(drop=True)
    result[label_col] = result[label_col].astype(int)
    result[score_column] = result[score_column].astype(float)
    result['weights'] = result['weights'].astype(int)
    return result[label_col], result[score_column], result['weights']

def adjust_predicts_pd(df, label_col, pred_col, groupby_col=None, index_column='timestamp', tol = 0):
    def helper(group):
        if not group[index_column].is_monotonic_increasing:
            raise ValueError("index_column must be monotonic")    
        pred= adjust_predicts(group[pred_col].values, group[label_col].values, calc_latency=False)
        group[pred_col] = pred
        return group
    df = df.groupby(groupby_col, group_keys=False).apply(helper)
    return df

def rolling_max_helper(group,features,win_size):
    edge_size = win_size//2
    extended_group = pd.concat([pd.DataFrame([[0]*len(features)]*edge_size, columns=features), group[features], pd.DataFrame([[0]*len(features)]*edge_size, columns=features)]).reset_index(drop=True)
    extended_group_max = extended_group.rolling(window=win_size, center=True).max()
    new_features = [f + '(max)' for f in features]
    if extended_group_max[edge_size:-edge_size].reset_index(drop=True).isnull().values.any():
        print('stop')
    group[new_features] = extended_group_max[edge_size:-edge_size].reset_index(drop=True).values
    return group

def tolerate_predicts(df,pred_col, label_col, index_column='timestamp', tol = 0, by_col='instance'):
    if not df[index_column].is_monotonic_increasing:
        raise ValueError("index_column must be monotonic")
    left_pred = pd.DataFrame(df[[index_column,pred_col,by_col]])
    left_pred = left_pred[left_pred['pred']==1]
    right_pred = left_pred.copy()
    left_pred[pred_col] = -left_pred[pred_col]
    left_pred.rename(columns={pred_col:pred_col+'_left'},inplace=True)
    right_pred.rename(columns={pred_col:pred_col+'_right'},inplace=True)
    label = pd.DataFrame(df[[index_column,label_col,by_col]])
    tol = pd.Timedelta(minutes=tol)
    merge_df = pd.merge_asof(label, left_pred, on=index_column, direction='backward',tolerance=tol,by=by_col)
    merge_df = pd.merge_asof(merge_df, right_pred, on=index_column, direction='forward',tolerance=tol,by=by_col)
    df[pred_col+'_direct'] = merge_df[pred_col+'_left'].combine_first(merge_df[pred_col+'_right']).fillna(0)
    df[pred_col]=  df[pred_col+'_direct'].abs()
    return df

def adjust_predicts(pred, label,
                    threshold=None,
                    score=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    label = np.asarray(label)
    latency = 0
    if pred is None and score is not None and threshold is not None:
        predict = score < threshold
        
    else:
        predict = pred
    actual = label
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(label)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, -1, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    predict = predict.astype(int)
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def sum_topk_scores(test_scores, topk=1):
    total_features = test_scores.shape[0]
    topk_indices = np.argpartition(test_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    total_topk_err_scores = []
    topk_err_score_map=[]
    total_topk_err_scores = np.sum(np.take_along_axis(test_scores, topk_indices, axis=0), axis=0)
    return topk_indices,total_topk_err_scores

def get_full_err_scores_single(args, df_re,df_gt,feature_map,data_filter={},  save2disk=False,by=None,flag='val'):
    scores = deepcopy(df_re.iloc[:,~df_re.columns.isin(args.features)])

    if by is None:
        scaler = RobustScaler()
        scores[feature_map] = get_deviation(df_re[feature_map],df_gt[feature_map])
        scores[feature_map] = scaler.fit_transform(scores[feature_map])
    else:
        scores[feature_map] = get_deviation(df_re[feature_map],df_gt[feature_map])
        min_period_per = scores.shape[0]/scores.shape[0]
        def groupby_helper(group,min_period_per):
            epsilon=1e-2
            min_period = int(min_period_per*group.shape[0])
            median = group.expanding(min_periods = min_period).median()
            q75 = group.expanding(min_periods = min_period).quantile(0.75)
            q25 = group.expanding(min_periods = min_period).quantile(0.25)
            median.iloc[:min_period] = median.iloc[min_period]
            q75.iloc[:min_period] = q75.iloc[min_period]
            q25.iloc[:min_period] = q25.iloc[min_period]
            return abs(group - median) / (q75 - q25 + epsilon)

        scores[feature_map] = scores.groupby(by, group_keys=False)[feature_map].apply(groupby_helper,min_period_per=min_period_per)
        scores = scores.reset_index(drop=True)
    return scores

def get_deviation(df_re,df_gt):
    df_deviation = np.abs(df_re.values - df_gt.values)
    if np.isnan(df_deviation).any():
        raise ValueError('nan value in df_deviation')
    return df_deviation



    



    
    return smoothed_err_scores, history

def cal_delay(df, tol_min):
    def helper(g):
        t_anomaly = g[g['label']==1].timestamp.max()
        t_border = t_anomaly + pd.Timedelta(minutes=tol_min)
        g['delay'] = -1
        mask = (g['timestamp'] > t_anomaly) & (g['timestamp'] <= t_border) & (g['pred']==1)
        if g.loc[mask].shape[0] == 0:
            return 0
        else:
            g.loc[mask,'delay'] = (g['timestamp'] - t_anomaly).dt.total_seconds()
            first_delay = g.loc[mask].iat[0, g.columns.get_loc('delay')]
            return first_delay

    df_copy = df.copy()
    delay = df_copy.groupby('instance', group_keys=False).apply(helper)
    return delay
    