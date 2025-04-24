import pandas as pd
import numpy as np
from util.data_artifact import Artifact
from util.normalization import scaler_wrapper
from util.time_range import first_hours, last_hours
from util.inject_anomalies import InjectAnomalies
import csv 
import os
import glob
import multiprocessing as mp
import json5
from sklearn.model_selection import train_test_split
from util.net_prep_old import get_feature_map, store_featurelist
class Data:
    def __init__(self, path,filter=None,args=None):
        self.path = path
        self.filter = filter
        self.args = args
        if hasattr(args, 'data_fillna'):
            self.df = self.read_data(fillna=args.data_fillna)
        else:
            self.df = self.read_data() 
        
    def read_data(self, fillna=True):
        if hasattr(self.args, 'model') and (self.args.model in ['tsmixer3', 'tsmixer4', 'tsmixer5']) and hasattr(self.args, 'job_log_path'):
            new_path = self.path.replace('.parquet','_with_job.parquet')
            if os.path.exists(new_path):
                self.path = new_path

        if self.path.split(".")[-1] == "csv":
            df_joined = (
                pd.read_csv(
                    self.path,
                )
            )
        elif self.path.split(".")[-1] == "parquet" or bool(glob.glob(os.path.join(self.path, '*.parquet'))):
            df_joined = (
                pd.read_parquet(
                    self.path,
                    engine="pyarrow"
                )
            )
        else:
            print("[!] File type not supported")
            return pd.DataFrame()

        if hasattr(self.args, 'timestamp_col'):
            df_joined.rename(columns={self.args.timestamp_col:'timestamp'}, inplace=True)
        if hasattr(self.args, 'instance_col'):
            df_joined.rename(columns={self.args.instance_col:'instance'}, inplace=True)
        if hasattr(self.args, 'job_id_col'):
            df_joined.rename(columns={self.args.job_id_col:'hnode_graph'}, inplace=True)
        if hasattr(self.args, 'label_col'):
            df_joined.rename(columns={self.args.label_col:'label'}, inplace=True)
        if hasattr(self.args, 'anomaly_source_col'):
            df_joined.rename(columns={self.args.anomaly_source_col:'anomaly_source'}, inplace=True)

        if 'label' not in df_joined.columns:
            df_joined['label'] = 0
        if 'status' in df_joined.columns:
            df_joined['status'] = df_joined['status'].fillna('Unknown')
        if 'instance' in df_joined.columns:
            df_joined['instance_id'] = df_joined['instance'].astype('category').cat.codes
        if 'GPU' in df_joined.columns:
            if df_joined[df_joined['label']==0]['GPU'].nunique()==1 or df_joined[df_joined['label']==0]['GPU'].nunique()==0:
                df_joined.loc[df_joined['label']==0, 'GPU'] = -1
            else:
                print('GPU error ID appears when the record is normal!')
        # parse timestamp
        if pd.api.types.is_datetime64_any_dtype(df_joined['timestamp']):
            pass
        elif pd.api.types.is_string_dtype(df_joined['timestamp']):
            unique_len_timestamp_str = df_joined['timestamp'].str.len().unique()
            if len(unique_len_timestamp_str) > 1:
                print('[WARN]: Timestamp column contains different length of string, please check the format')
                df_joined['timestamp'] = df_joined['timestamp'].str[:19]
                print('[WARN]: Timestamp column has been truncated to 19 characters, in order to fit the format: YYYY-MM-DDTHH:MM:SS')
            elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 19):
                pass
            elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 32):
                df_joined['timestamp'] = df_joined['timestamp'].str[:26]
            elif len(unique_len_timestamp_str) == 1 and (unique_len_timestamp_str[0] == 26):
                pass
            else:
                raise ValueError('Timestamp column contains invalid string length')
            
            df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])
        elif pd.api.types.is_integer_dtype(df_joined['timestamp']):
            if hasattr(self.args, 'dataset') and self.args.dataset == 'prodigy_mini':
                df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'],unit='s')
                df_joined['anomaly_source'] = 0
            else:
                df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])
        else:
            raise ValueError('Timestamp column must be datetime64 or string or int')
        
        df_joined= df_joined.sort_values(by=["timestamp"])
        if self.filter is not None and self.filter != {}:
            df_joined = filter_df(df_joined, self.filter)
        df_joined= self.remove_original_index(df_joined)
        df_joined = df_joined.reset_index(drop=True)
        # check whether contains Nan
        nan_columns = df_joined.columns[df_joined.isna().any()].tolist()
        if len(nan_columns)>0:
            # print('[WARN]: Dataset contains Nan, please check the following columns:')
            # print('[Nan columns]:',nan_columns)
            if fillna:
                # print('[WARN]: Nan in Dataset will be filled with 0.')
                df_joined = df_joined.fillna(0) # fill nan with 0
        
        return df_joined
    
    def set_abnormal(self, time_ranges):
        return set_val_by_time_range(self.df, time_ranges)

    def by_time_range(self, time_ranges):
        return slice_df_by_time_range(self.df, time_ranges)
        
    def remove_original_index(self, df):
        """
        Remove the original index column.
        """
        if 'index' in df.columns:
            df= df.drop('index', axis=1)
        if 'Unnamed: 0' in df.columns:
            df= df.drop('Unnamed: 0', axis=1)
        return df

def make_artifact2(df, time_ranges,farmnodes='random',num_faramnodes=2,feature='random',feature_map=None,inject_num=5,attack_type='spikes',logpath=None,skip_constant=True):
    df['fine_label']=0
    np.random.seed() # set seed to random
    verbose = True
    feature_list = []
    farmnodes_list = []
    fmp_temp = feature_map.copy()
    all_instances_temp = df['instance'].unique().tolist()
    feat_counter = inject_num
    df['anomaly_source'] = None
    while feat_counter >0: # In one iteration, inject anomalies for one feature on multiple farmnodes
        skip_cur_feat_flag = False
        feat_counter -= 1
        if  fmp_temp==None or len(fmp_temp)==0:
            raise ValueError('[Error] anomaly injector tries to inject {} features, but fails.'.format(inject_num))
        feature  = np.random.choice(fmp_temp)
        fmp_temp.remove(feature)
        if df.loc[:,feature].unique().shape[0]==1 and skip_constant==True:
            feat_counter += 1
            continue

        def helper(group):
            diff = group[feature].diff()
            diff.fillna(0, inplace=True)
            if diff.max() > 0:
                return True
            else:
                return False
        cur_active_instances = df.groupby('instance').apply(helper)
        cur_active_instances = cur_active_instances[cur_active_instances==True].index.values
        cur_instances_temp = np.intersect1d(all_instances_temp,cur_active_instances)
        
        T_with_anomaly_list = []
        anomaly_labels_list = []
        farmnodes_perfeat_list =[]
        masks = []
        num_faramnodes_counter = num_faramnodes
        while num_faramnodes_counter >0 and len(cur_instances_temp)>0 and not skip_cur_feat_flag:
            num_faramnodes_counter -= 1
            farmnodes = np.random.choice(cur_instances_temp,1,replace=False).tolist()
            cur_instances_temp = list(set(cur_instances_temp)-set(farmnodes)) # without duplicate
            mask = get_mask_by_time_range(df, time_ranges,farmnodes=farmnodes[0])
            if df.loc[mask,feature].unique().shape[0]==1 and skip_constant==True:
                if verbose: print('[Warning] Inject anomaly {} on {}: Constant Feature'.format(feature,farmnodes[0]))
                continue
            success_inject = False
            inject_trial_counter = 0
            while(success_inject==False and inject_trial_counter<100):
                anomalyObj = InjectAnomalies(
                                    random_state=np.random.randint(10000), 
                                    verbose=False, 
                                    max_window_size=128, 
                                    min_window_size=8,
                                    noise_dist='exp') #SET norm or exp  
                data_std = max(np.std(df.loc[:,feature].values), 0.01)
                T_with_anomaly, anomaly_sizes, anomaly_labels = anomalyObj.inject_anomalies(df.loc[mask,feature].values.T,
                                                                                scale=6*data_std, 
                                                                                anomaly_type=attack_type, #SET spikes, contextual, flip, speedup, noise, cutoff, scale, wander, average.
                                                                                random_parameters=False,
                                                                                anomaly_size_type='mae',
                                                                                max_anomaly_length=4,
                                                                                constant_type='quantile', 
                                                                                speed=6,
                                                                                amplitude_scaling= 2)
                if anomaly_labels.sum()>=(df.loc[mask,feature].shape[0]//20) and df.loc[mask,feature].shape[0]>10: # at least 10 anomalies
                    success_inject = True
                else:
                    inject_trial_counter +=1
            if success_inject==False:
                if verbose: print('[Warning] Inject anomaly {}: Anomalous length is less than 10'.format(feature))
                continue
            T_with_anomaly_list.append(T_with_anomaly)
            anomaly_labels_list.append(anomaly_labels)
            masks.append(mask)
            farmnodes_perfeat_list.append(farmnodes[0])

        if (skip_cur_feat_flag== True and skip_constant==True) or len(masks)!=num_faramnodes:
            if verbose: print('[Warning] Inject anomaly {}: Injected computational nodes are less than {}'.format(feature,num_faramnodes))
            feat_counter += 1
            continue
        else:
            feature_list.append(feature)
            farmnodes_list.append(farmnodes_perfeat_list)
            all_instances_temp= list(set(all_instances_temp)-set(farmnodes))

        for i in range(len(masks)):
            mask = masks[i]
            T_with_anomaly = T_with_anomaly_list[i]
            if T_with_anomaly.shape[0]==1:
                T_with_anomaly = T_with_anomaly[0]
            anomaly_labels = anomaly_labels_list[i]
            if df.loc[mask,feature].shape[0]==T_with_anomaly.shape[0]:
                df.loc[mask,feature] = T_with_anomaly
                df.loc[mask,'fine_label'] = df.loc[mask,'label'].values | anomaly_labels 
                df.loc[mask,'label'] = 1 
                df.loc[mask,'anomaly_source'] = feature
            else: # speedup would make the shape not aligned, repeat T_with_anomaly to match the shape of df.loc[mask,feature]
                T_with_anomaly = np.tile(T_with_anomaly, df.loc[mask,feature].shape[0]//T_with_anomaly.shape[0]+1)[:df.loc[mask,feature].shape[0]]
                anomaly_labels = np.tile(anomaly_labels, df.loc[mask,feature].shape[0]//anomaly_labels.shape[0]+1)[:df.loc[mask,feature].shape[0]]
                df.loc[mask,feature] = T_with_anomaly
                df.loc[mask,'fine_label'] = df.loc[mask,'label'].values | anomaly_labels
                df.loc[mask,'label'] = 1
                df.loc[mask,'anomaly_source'] = feature
        anomaly_part = df[(df['label']==1)&(df['anomaly_source']==feature)]
        if anomaly_part['instance'].dropna().nunique()!= num_faramnodes:
            raise ValueError('[Error] anomaly injector injects farmnodes less than {} for feature {}.'.format(num_faramnodes,feature))

    # duplicated check
    anomaly_part = df[df['label']==1]
    if len(anomaly_part['anomaly_source'].dropna().unique())!= inject_num:
        real_inject_feat_num=len(df['anomaly_source'].dropna().unique())
        raise ValueError('[Error] anomaly injector injects{}, less than {}.'.format(real_inject_feat_num,inject_num))
    if len(anomaly_part['instance'].dropna().unique())!= (num_faramnodes*inject_num):
        real_farmnodes_perfeat_list_debug= anomaly_part.groupby('anomaly_source')['instance'].unique()
        print(real_farmnodes_perfeat_list_debug.astype('string').rename('num_instance').to_markdown())
        print('[Error] anomaly injector injects farmnodes less than {}.'.format(num_faramnodes))
    
    np.random.seed(5) # set seed back to 5
    if logpath==None:         # log attack time range and feature to file
        logpath = './attack_time_range.csv'
    with open(logpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start_time', 'end_time','feature','farmnodes'])
        for start_time, end_time in time_ranges:
            for feature,farmnodes in zip(feature_list,farmnodes_list):
                writer.writerow([start_time, end_time,feature,farmnodes])
    return df

def set_val_by_time_range(df, time_ranges, column='label',val=1, farmnodes=None):
    if column not in df.columns:
        df[column] = 0
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df['timestamp'] <= end_time)
        elif end_time == '':
            mask = (df['timestamp'] >= start_time)
        else:
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        if farmnodes!=None:
            mask = mask & (df['instance'].isin(farmnodes))
        # Get the rows that fall within the current range
        if isinstance(val, list):
            df.loc[mask,column] = df.loc[mask,column].apply(lambda x: val)
        elif isinstance(val, int) or isinstance(val, float) or isinstance(val, str):
            df.loc[mask,column] = val
    return df

def slice_df_by_time_range(df, time_ranges,timestamp_col='timestamp'): 
    # Create an empty dataframe to hold the sliced data
    sliced_df = pd.DataFrame()
    # Loop over each timestamp range
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df[timestamp_col] <= end_time)
        elif end_time == '':
            mask = (df[timestamp_col] >= start_time)
        else:
            mask = (df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)
        # Get the rows that fall within the current range
        df_range = df.loc[mask]
        # Append the rows to the sliced dataframe
        sliced_df = pd.concat([sliced_df, df_range])
        # print Nan columns
        nan_columns = sliced_df.columns[sliced_df.isna().any()].tolist()
    sliced_df = sliced_df.reset_index(drop=True)
    return sliced_df

def get_mask_by_time_range(df, time_ranges,farmnodes=None):
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError('Start time must be less than end time')
        if start_time == '':
            mask = (df['timestamp'] <= end_time)
        elif end_time == '':
            mask = (df['timestamp'] >= start_time)
        else:
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        if farmnodes!=None:
            if isinstance(farmnodes, list):
                mask = mask & (df['instance'].isin(farmnodes))
            elif isinstance(farmnodes, str):
                mask = mask & (df['instance']==farmnodes)
            else:
                raise ValueError('farmnodes must be a list or a string')
    return mask

def filter_df(df, filter):
    for key in filter.keys():
        if key not in df.columns:
            print("[!] No column: " + key)
            return pd.DataFrame()
        # check if params[key] is a list
        if isinstance(filter[key], list):
            df = df[df[key].isin(filter[key])]
        # check if params[key] is a string
        if isinstance(filter[key], str):
            # check if params[key] is a regex expression
            if filter[key].startswith('/') and filter[key].endswith('.*/'):
                df = df[df[key].str.startswith(filter[key][1:-3])]
            else:
                df = df[df[key] == filter[key]]
    return df.reset_index(drop=True)

def build_attack(test, args):
    print("------------------injecting artifact------------------")
    cgt_savepath = os.path.dirname(args.paths['test_re'])+"/gt_test_artifact.csv"
    if not os.path.isfile(cgt_savepath):
        print("------------------initialize artifact------------------")
        logpath = os.path.dirname(args.paths['test_re'])+"/attack_time_range.csv"
        if hasattr(args, 'test_time_ranges'):
            test_time_ranges= args.test_time_ranges
        elif hasattr(args, 'test_ratio') or hasattr(args, 'val_ratio'):
            test = test
            test_time_ranges = [[test['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'), test['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')]]
        skip_constant = True
        if args.dataset =='jlab':
            inject_num = 5
            cpu_feature_map = args.features[:8] # cpu feature
            mem_feature_map = args.features[8:55] # mem feature
            disk_feature_map = args.features[55:] # disk feature
            if args.save_subdir.endswith('cpu'):
                select_feature_map=cpu_feature_map
            elif args.save_subdir.endswith('mem'):
                select_feature_map=mem_feature_map
            elif args.save_subdir.endswith('disk'):
                select_feature_map=disk_feature_map
            elif args.save_subdir.endswith('mem_neighbors'):
                g = pd.read_csv(args.paths['graph_csv'])
                g = g.sort_values(by=["value"],ascending=False)
                g = g[g['source']!=g['destination']]
                features_with_neighbors = g['source'].unique().tolist()+g['destination'].unique().tolist()
                features_with_neighbors = list(map(lambda x: args.features[x], list(set(features_with_neighbors))))
                features_with_neighbors = [x for x in features_with_neighbors if x in mem_feature_map]
                select_feature_map=features_with_neighbors
            else:
                raise ValueError('save_subdir must end with cpu, mem or disk')
        elif args.dataset =='olcfcutsec':
            inject_num = 2
            wholepower_feature_map = ['ps0_input_power','ps1_input_power']
            cpu_feature_map = [f for f in args.features if f.startswith('p0_temp') or f.startswith('p1_temp') or f.startswith('p0_power') or f.startswith('p1_power')]
            gpu_feature_map = [f for f in args.features if f not in cpu_feature_map and f not in wholepower_feature_map]
            if args.save_subdir.endswith('cpu'):
                select_feature_map=cpu_feature_map
            elif args.save_subdir.endswith('gpu'):
                select_feature_map=gpu_feature_map
            elif args.save_subdir.endswith('wholepower'):
                select_feature_map=wholepower_feature_map
            else:
                raise ValueError('save_subdir must end with cpu, gpu or wholepower')

        args.attack_type = args.save_subdir.split('_')[0]
        test = make_artifact2(test,last_hours(test_time_ranges,1),feature='random',feature_map=select_feature_map,inject_num=inject_num,attack_type=args.attack_type,logpath=logpath,skip_constant=skip_constant)
        test.to_csv(os.path.dirname(args.paths['test_re'])+"/gt_test_artifact.csv")
    else:
        print("------------------artifact already injected------------------")
        status = None
        if 'status' in test.columns:
            status = test[['timestamp','instance','status']]
        test = Data(os.path.dirname(args.paths['test_re'])+"/gt_test_artifact.csv").df
        test.rename(columns={'attack':'label'}, inplace=True)
        if 'status' not in test.columns and status is not None:
            test = pd.merge(test, status, on=['timestamp', 'instance'], how='left')
    return test

def sort_wrapper(df,by_column=["instance","timestamp"]):
    df.sort_values(by=by_column,inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def uni_timestamp(df, gap, tolerance):
    init_time = df['timestamp'][0]
    df['uni_timestamp'] = df['timestamp'].apply(lambda x: (x-init_time).total_seconds()//gap)
    return df

def fill_missing_instance(df):
    df = uni_timestamp(df, gap=60, tolerance=0.1)
    all_timestamps = df['uni_timestamp'].unique()
    all_instances = df['instance'].unique()
    mux = pd.MultiIndex.from_product([all_timestamps, all_instances], names=['uni_timestamp', 'instance'])
    df_full = pd.DataFrame(index=mux).reset_index()
    df_full = pd.merge(df_full, df, how='left', on=['uni_timestamp', 'instance'])
    return df_full

def hardcode_args(args,type):
    if type == 'real_world':
        args.test_time_ranges=[
            ['2023-05-23 06:00:00', '2023-05-23 17:00:00']
            ]


class Dataset:
    def __init__(self, args):
        with open(args.paths['dataset']+"/data_config.json") as json_file:
            data_args = json5.load(json_file)
            for key in data_args:
                if key in args.__dict__ and (key == 'test_file' or key=='whole_file'):
                    # print(f"[WARN]: {key}:{getattr(args,key)} already exists in args, skip loading from data_config.json")
                    pass
                else:
                    setattr(args, key, data_args[key])
            hardcode_args(args, args.save_subdir)


        self.train, self.test, self.val = self.load_data(args)
        args.features = get_feature_map(args.paths['dataset'])
        if args.features == None:
            args.features = store_featurelist(args.paths['dataset'], self.train)

        if hasattr(args,'scaler_only_fit') and args.scaler_only_fit:
            self.train, self.test, self.val, self.scaler = scaler_wrapper(self.train, self.test, self.val, args.features, args.scaler, only_fit=True)
        else:
            self.train, self.test, self.val, self.scaler = scaler_wrapper(self.train, self.test, self.val, args.features, args.scaler)
        if args.attack: 
            if hasattr(args, 'retain_beforeattack') and args.retain_beforeattack:
                self.test_before_attack = self.test.copy()
            self.test = build_attack(self.test, args)

        self.train = sort_wrapper(self.train,by_column=["timestamp","instance"])
        self.test = sort_wrapper(self.test,by_column=["timestamp","instance"])
        self.val = sort_wrapper(self.val,by_column=["timestamp","instance"])

    def load_data(self, args):
        dataset_path = args.paths['dataset']
        # check if dataset_path is a folder
        if hasattr(args, 'whole_file'):
            data = Data(dataset_path+'/'+args.whole_file, filter=args.data_filter, args=args)

            if hasattr(args, 'attack_time_ranges') and hasattr(args, 'train_time_ranges') and hasattr(args, 'test_time_ranges') and hasattr(args, 'val_time_ranges'):
                data.set_abnormal(args.attack_time_ranges) # set abnormal time range
                train = data.by_time_range(args.train_time_ranges)
                test = data.by_time_range(args.test_time_ranges)
                val = data.by_time_range(args.val_time_ranges)
                del data # delete original data to save memory
                return train, test, val

            elif hasattr(args, 'train_time_ranges') and hasattr(args, 'test_time_ranges') and hasattr(args, 'val_time_ranges'):
                train = data.by_time_range(args.train_time_ranges)
                test = data.by_time_range(args.test_time_ranges)
                val = data.by_time_range(args.val_time_ranges)
                del data # delete original data to save memory
                return train, test, val

            elif hasattr(args, 'train_ratio') and hasattr(args, 'test_ratio') and hasattr(args, 'val_ratio'):
                remain_df, test = train_test_split(data.df, test_size=args.test_ratio, shuffle=False)
                train, val = train_test_split(remain_df, test_size=args.val_ratio/(args.train_ratio + args.val_ratio), shuffle=False)
                del data # delete original data to save memory
                return train, test, val

        elif hasattr(args, 'train_file') and hasattr(args, 'test_file'):
            train = Data(dataset_path +'/'+ args.train_file, filter=args.data_filter, args=args)
            test = Data(dataset_path +'/'+ args.test_file, filter=args.data_filter, args=args)
            if hasattr(args, 'val_ratio') and not hasattr(args,'val_time_ranges'):
                train = train.df
                test = test.df
                train, val = train_test_split(train, test_size=args.val_ratio, shuffle=False)
                return train, test, val
            elif hasattr(args, 'val_ratio') and hasattr(args,'val_time_ranges'):
                
                train = train.df
                test = test.df
                train, _ = train_test_split(train, test_size=args.val_ratio, shuffle=False)
                whole = pd.concat([train, test])
                val = slice_df_by_time_range(whole, args.val_time_ranges)
                return train, test, val
            else:
                return train, test, pd.DataFrame()

    def align_instance(self,args):
        args.hnodes = self.train['instance'].unique()
        args.hnode_num = len(args.hnodes)
        self.test = self.test[self.test['instance'].isin(args.hnodes)]
        self.val = self.val[self.val['instance'].isin(args.hnodes)]