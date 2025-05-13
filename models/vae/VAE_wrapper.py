from models.vae.data_pipeline import DataPipeline
from models.vae.vae_layer import VAE
import json
from tsfresh.feature_extraction import settings
import os
import tensorflow as tf
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, classification_report
import numpy as np

def generate_tsindex(df, timestamp_column='timestamp', instance_id_column='instance_id',window_size=1, tsindex_column='job_id'):
    job_id = 0
    
    def assign_job_id(group):
        nonlocal job_id
        group = group.sort_values(by=timestamp_column).reset_index(drop=True)
        group[tsindex_column] = (group.index // window_size) + job_id
        job_id += (len(group) + window_size - 1) // window_size
        return group
    
    df = df.groupby(instance_id_column).apply(assign_job_id).reset_index(drop=True)
    return df

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
    

class VAE_wrapper(object):
    def __init__(self, args, data, tensorboard=False):
        self.args = args
        self.data = data
        self.prepare_data(extract_pre_selected_features=False)
        self._setup_model()
    
    def prepare_data(self,extract_pre_selected_features=True):

        def get_instance_level_label(df,label_column='label',timestamp_column='timestamp', instance_column='instance'):
            df['job_id'] = df['job_id'].astype(str)
            df['instance'] = df['instance'].astype(str)
            return_df = df.groupby(['job_id','instance']).agg(label=("label",'max'))
            return return_df
        
        pre_selected_features_filename = 'data/prodigy_mini/content.json'
        if pre_selected_features_filename is not None:
            with open(pre_selected_features_filename, "r") as fp:
                selected_features_json = json.load(fp)
            fe_selected_features = selected_features_json['tsfresh_column_names']
            print("The previously selected features will be used")


        if os.path.exists(os.path.dirname(self.args.paths['train_re'])+'/train_vae_gt.csv'):
            self.x_train_scaled = pd.read_csv(os.path.dirname(self.args.paths['train_re'])+'/train_vae_gt.csv')
            self.y_train = pd.read_csv(os.path.dirname(self.args.paths['train_re'])+'/train_vae_gt_label.csv')
            self.x_train_scaled.set_index(['job_id','instance'],inplace=True)
            self.y_train.set_index(['job_id','instance'],inplace=True)
        else:
            if not hasattr(self.args, 'hgdn_graph_col'):
                self.data.train = generate_tsindex(self.data.train, timestamp_column='timestamp', instance_id_column='instance_id',window_size=6*60, tsindex_column='hnode_graph')
            self.data.train.rename(columns={'hnode_graph':'job_id'},inplace=True)
            x_train = self.data.train
            y_train = get_instance_level_label(x_train)
            pipeline = DataPipeline()
            if extract_pre_selected_features:
                x_train_fe = pipeline.tsfresh_generate_features(x_train[['instance','timestamp','job_id']+self.args.features], fe_config=None, kind_to_fc_parameters=fe_selected_features)
            else:
                x_train_fe = pipeline.tsfresh_generate_features(x_train[['instance','timestamp','job_id']+self.args.features], fe_config="minimal")
            if y_train is not None:
                y_train = y_train.loc[x_train_fe.index]
            assert all(y_train.index == x_train_fe.index)
            x_train_fe.to_csv(os.path.dirname(self.args.paths['train_re'])+'/train_vae_gt.csv')   
            y_train.to_csv(os.path.dirname(self.args.paths['train_re'])+'/train_vae_gt_label.csv')
            self.x_train_scaled = x_train_fe
            self.y_train = y_train


        if os.path.exists(os.path.dirname(self.args.paths['test_re'])+'/test_vae_gt.csv'):
            self.x_test_scaled = pd.read_csv(os.path.dirname(self.args.paths['test_re'])+'/test_vae_gt.csv')
            self.y_test = pd.read_csv(os.path.dirname(self.args.paths['test_re'])+'/test_vae_gt_label.csv')
            self.x_test_scaled.set_index(['job_id','instance'],inplace=True)
            self.y_test.set_index(['job_id','instance'],inplace=True)
        else:
            if not hasattr(self.args, 'hgdn_graph_col'):
                self.data.test = generate_tsindex(self.data.test, timestamp_column='timestamp', instance_id_column='instance_id',window_size=6*60, tsindex_column='hnode_graph')
            self.data.test.rename(columns={'hnode_graph':'job_id'},inplace=True)
            x_test = self.data.test
            y_test = get_instance_level_label(x_test)
            pipeline = DataPipeline()
            if extract_pre_selected_features:
                x_test_fe = pipeline.tsfresh_generate_features(x_test[['instance','timestamp','job_id']+self.args.features], fe_config=None, kind_to_fc_parameters=fe_selected_features)
            else:
                x_test_fe = pipeline.tsfresh_generate_features(x_test[['instance','timestamp','job_id']+self.args.features], fe_config="minimal")
            if y_test is not None:
                y_test = y_test.loc[x_test_fe.index]
            assert all(y_test.index == x_test_fe.index)            
            x_test_fe.to_csv(os.path.dirname(self.args.paths['test_re'])+'/test_vae_gt.csv')
            y_test.to_csv(os.path.dirname(self.args.paths['test_re'])+'/test_vae_gt_label.csv')
            self.x_test_scaled = x_test_fe
            self.y_test = y_test


        if len(self.x_test_scaled.columns) < len(self.x_train_scaled.columns):
            self.x_train_scaled = self.x_train_scaled[self.x_test_scaled.columns]
        elif len(self.x_test_scaled.columns) > len(self.x_train_scaled.columns):
            self.x_test_scaled = self.x_test_scaled[self.x_train_scaled.columns]
        self.x_train_scaled = self.x_train_scaled[self.x_test_scaled.columns]
        assert all(self.x_train_scaled.columns == self.x_test_scaled.columns)
        self.x_test_scaled = self.x_test_scaled.loc[self.y_test.index]

        if self.x_test_scaled is not None:
            output_dir =os.path.dirname(self.args.paths['train_re'])
            self.x_train_scaled, self.x_test_scaled = DataPipeline().scale_data(self.x_train_scaled, self.x_test_scaled, save_dir=output_dir)
        else:
            output_dir =os.path.dirname(self.args.paths['train_re'])
            self.x_train_scaled, self.x_test_scaled = DataPipeline().scale_data(self.x_train_scaled, None, save_dir=output_dir)

    def _setup_model(self):
        input_dim = self.x_train_scaled.shape[1]
        intermediate_dim = int(input_dim / 2)
        latent_dim = int(input_dim / 3)

        self.model =  VAE(
                name="model",
                input_dim=input_dim,
                intermediate_dim=intermediate_dim,
                latent_dim=latent_dim,
                learning_rate=1e-4
            )


    def train(self):
        output_dir =os.path.dirname(self.args.paths['train_re'])
        self.model.fit(
            x_train=self.x_train_scaled,
            epochs=1000,
            batch_size=32,
            validation_split=0.1,
            save_dir=output_dir,
            verbose=0
        )



    def score(self):
        if self.args.threshold_per is not None and self.args.threshold_per != -1:
            threshold_per_list = [self.args.threshold_per]
        else:
            print('>>>   search best f1 score')
            threshold_per_list = [0.9999]

        f1_list = []
        prec_list = []
        recall_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        tpr_list = []
        fpr_list = []
        if hasattr(self.args,'tol_min'):
            delay_list = []
            num_delay_list = []

        for threshold_per in threshold_per_list:
            if self.x_test_scaled is not None:
                assert all(self.x_test_scaled.columns == self.x_train_scaled.columns)
                assert all(self.x_test_scaled.index == self.y_test.index)

                y_pred_test, x_test_recon_errors, x_train_recon_errors = self.model.best_predict_anomaly(self.x_train_scaled,self.x_test_scaled,self.y_test['label'].values,threshold_per_list=[threshold_per])
                print(f"Selected threshold percentage: {self.model.threshold_per}")

            if self.y_test is not None:
                self.y_test_copy = self.y_test.copy()
                self.y_test_copy['pred'] = y_pred_test
                self.y_test_copy['recon_errors'] = x_test_recon_errors
                
                if not hasattr(self.args, 'hgdn_graph_col'):
                    testdata = generate_tsindex(self.data.test.copy(), timestamp_column='timestamp', instance_id_column='instance_id',window_size=6*60, tsindex_column='job_id')
                else:
                    testdata = self.data.test.copy()
                    testdata.rename(columns={'hnode_graph':'job_id'},inplace=True)

                self.y_test_copy = self.y_test_copy.reset_index()

                self.y_test_copy['instance']= self.y_test_copy['instance'].astype(str)
                self.y_test_copy['job_id']= self.y_test_copy['job_id'].astype(int)
                testdata['instance']= testdata['instance'].astype(str)
                testdata['job_id']= testdata['job_id'].astype(int)

                testdata = testdata.merge(self.y_test_copy, on=['job_id', 'instance'], how='left', suffixes=('', '_ytest'))

                f1 = f1_score(testdata['label'].values, testdata['pred'].values,average='macro')
                precision = precision_score(testdata['label'].values, testdata['pred'].values,average='macro')
                recall = recall_score(testdata['label'].values, testdata['pred'].values,average='macro')
                if testdata['label'].nunique() == 1:
                    auroc = -1
                else:
                    auroc = roc_auc_score(testdata['label'].values, testdata['recon_errors'].values,average='macro')
                tp = testdata[(testdata['pred']==1)& (testdata['label']==1)].shape[0]
                fn = testdata[(testdata['pred']==0)& (testdata['label']==1)].shape[0]
                fp = testdata[(testdata['pred']==1)& (testdata['label']==0)].shape[0]
                tn = testdata[(testdata['pred']==0)& (testdata['label']==0)].shape[0]
                if tp+fn == 0:
                    tpr = -1
                    fpr = -1
                else:
                    tpr = tp/(tp+fn)
                    fpr = fp/(fp+tn)

                print(f"---Timestamp level---")
                print(f"F1 score: {f1}")
                print(f"Precision score: {precision}")
                print(f"Recall score: {recall}")
                print(f"AUROC score: {auroc}")
                print(f"True Positive: {tp}")
                print(f"False Negative: {fn}")
                print(f"False Positive: {fp}")
                print(f"True Negative: {tn}")
                print(f"True Positive Rate: {tpr}")
                print(f"False Positive Rate: {fpr}")


                f1_list.append(f1)
                prec_list.append(precision)
                recall_list.append(recall)
                tp_list.append(tp)
                fp_list.append(fp)
                tn_list.append(tn)
                fn_list.append(fn)
                tpr_list.append(tpr)
                fpr_list.append(fpr)
                if hasattr(self.args,'tol_min'):
                    delays = cal_delay(testdata, self.args.tol_min)
                    num_delay = delays[delays!=0].shape[0]
                    if num_delay == 0:
                        average_delay = -1
                    else:
                        average_delay = delays[delays!=0].mean()
                    delay_list.append(average_delay)
                    num_delay_list.append(num_delay)

        f1_list = np.array(f1_list)
        prec_list = np.array(prec_list)
        recall_list = np.array(recall_list)
        tp_list = np.array(tp_list)
        fp_list = np.array(fp_list)
        tn_list = np.array(tn_list)
        fn_list = np.array(fn_list)
        tpr_list = np.array(tpr_list)
        fpr_list = np.array(fpr_list)
        threshold_per_list = np.array(threshold_per_list)
            
        save_df = pd.DataFrame(columns=['threshold_per','f1','precision','recall','tp','fp','tn','fn','tpr','fpr'])
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
        if hasattr(self.args,'tol_min'):
            save_df['delay'] = np.array(delay_list)
            save_df['num_delay'] = np.array(num_delay_list)


        Path(os.path.dirname(self.args.paths['test_re'])+'/scores/anomaly_detect/').mkdir(parents=True, exist_ok=True)
        save_df.to_csv(os.path.dirname(self.args.paths['test_re'])+'/scores/anomaly_detect/eval_threshold.csv',index=False)



    def load(self):
        if not hasattr(self.args, 'load_model_path') or self.args.load_model_path == '':
            return True
        elif not os.path.exists(self.args.load_model_path):
            print("Model path does not exist:", self.args.load_model_path)
            return True
        else:
            self.model.load_model_weights(self.args.load_model_path)