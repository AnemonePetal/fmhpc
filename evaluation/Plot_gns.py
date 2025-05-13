import sys
import os 
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import pandas as pd
import numpy as np
from util.Data import slice_df_by_time_range
from util.net_prep import convert_adj2edges
from matplotlib import pyplot as plt
from util.Feature_line_plot import single_line_plot,twin_line_plot
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42	

import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def extract_number(s):
    return float(''.join([ch for ch in s if ch.isdigit() or ch == '.']))

def read_graph_1(graph_path):
    df_dicts = {}
    df_std_dicts = {}

    dirs = os.listdir(graph_path)
    for dir_name in dirs:
        dir_path = os.path.join(graph_path, dir_name)
        if os.path.isdir(dir_path):
            filepath = os.path.join(dir_path, "profile_graph_adj.csv")
            filepath_std = os.path.join(dir_path, "profile_graph_std.csv")
            if os.path.exists(filepath):
                df_dicts[dir_name] = pd.read_csv(filepath)
                df_dicts[dir_name] = pd.DataFrame(convert_adj2edges(df_dicts[dir_name]),columns=['source','destination','value'])
                df_dicts[dir_name]['source'] = df_dicts[dir_name]['source'].apply(lambda x: int(x))
                df_dicts[dir_name]['destination'] = df_dicts[dir_name]['destination'].apply(lambda x: int(x))
                df_std_dicts[dir_name] = pd.read_csv(filepath_std)
                df_std_dicts[dir_name] = pd.DataFrame(convert_adj2edges(df_std_dicts[dir_name]),columns=['source','destination','value'])
                df_std_dicts[dir_name]['source'] = df_std_dicts[dir_name]['source'].apply(lambda x: int(x))
                df_std_dicts[dir_name]['destination'] = df_std_dicts[dir_name]['destination'].apply(lambda x: int(x))
    return df_dicts,df_std_dicts

def read_graph_2(graph_path):
    df_dicts = {}
    df_std_dicts = {}
    dirs = os.listdir(graph_path)
    for dir_name in dirs:
        dir_path = os.path.join(graph_path, dir_name)
        if os.path.isdir(dir_path):
            filepath = os.path.join(dir_path, "adj")
            filepath_std = os.path.join(dir_path, "std")
            if os.path.exists(filepath):
                df_dicts[dir_name] = pd.read_csv(filepath)
                df_dicts[dir_name] = pd.DataFrame(convert_adj2edges(df_dicts[dir_name]),columns=['source','destination','value'])
                df_dicts[dir_name]['source'] = df_dicts[dir_name]['source'].apply(lambda x: int(x))
                df_dicts[dir_name]['destination'] = df_dicts[dir_name]['destination'].apply(lambda x: int(x))
                df_std_dicts[dir_name] = pd.read_csv(filepath_std)
                df_std_dicts[dir_name] = pd.DataFrame(convert_adj2edges(df_std_dicts[dir_name]),columns=['source','destination','value'])
                df_std_dicts[dir_name]['source'] = df_std_dicts[dir_name]['source'].apply(lambda x: int(x))
                df_std_dicts[dir_name]['destination'] = df_std_dicts[dir_name]['destination'].apply(lambda x: int(x))
    return df_dicts,df_std_dicts


def reshape_df_duration(df, freq="1h"):
    def segment_time(row):
        segments = []
        begin_time = row['begin_time']
        end_time = row['end_time']
        
        while end_time > begin_time:
            next_hour = (begin_time + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if next_hour > end_time:
                next_hour = end_time
            segments.append((begin_time, next_hour))
            begin_time = next_hour
        return segments
    segmented_times = df.apply(segment_time, axis=1)
    segmented_data = []
    for idx, segments in segmented_times.iteritems():
        for seg in segments:
            segmented_data.append({
                'node_name': df.loc[idx, 'node_name'],
                'allocation_id': df.loc[idx, 'allocation_id'],
                'begin_time': seg[0],
                'end_time': seg[1],
                'begin_yymm': df.loc[idx, 'begin_yymm'],
                'end_yymm': df.loc[idx, 'end_yymm'],
            })

    segmented_df = pd.DataFrame(segmented_data)
    return segmented_df

def custom_resampler(group):
    num_jobs= group['allocation_id'].nunique()
    all_jobs = group['allocation_id'].value_counts(sort=True).to_json()
    sum_duration = (group['end_time'] - group['begin_time']).sum()
    if group['allocation_id'].nunique() == 1:
        flag_single_job = 1
    elif group['allocation_id'].nunique() == 0:
        flag_single_job = -1
    else:
        flag_single_job = 0
    return pd.DataFrame({'num_jobs': [num_jobs], 'all_jobs': [all_jobs], 'flag_single_job': [flag_single_job],'sum_duration':[sum_duration]})

def plot_line(df_slurm,filename='null'):
    fig, ax = plt.subplots(figsize=(15, 5))
    df_slurm = df_slurm.reset_index()
    twin_line_plot(ax, df_slurm, ['nei_similarity','sum_duration'],x_col='index',zoom=False)
    plt.savefig("data/olcfcutsec/figs/" +filename + "_sim.png")

def annotation_line( ax, xmin, xmax, y, text, ytext=0, linecolor='black', linewidth=1, fontsize=12 ):
    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|', 'color':linecolor, 'linewidth':linewidth})
    ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '<->', 'color':linecolor, 'linewidth':linewidth})
    xcenter = xmin + (xmax-xmin)/2
    if ytext==0:
        ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20
    ax.annotate( text, xy=(xcenter,ytext), ha='center', va='center', fontsize=fontsize)

def tuple_minus(t1, t2):
    list1 = list(t1)
    list2 = list(t2)
    for item in list2:
        if item in list1:
            list1.remove(item)
    return tuple(list1)


def plot_line2(df_slurm_segs, df_sim0,farmnodes,savepath):
    df_sim = df_sim0.copy()
    df_slurm_segs_allnode = df_slurm_segs[df_slurm_segs['node_name'].isin(farmnodes)]
    fig, ax = plt.subplots(2*len(farmnodes),1,figsize=(15, 2.2*len(farmnodes)),sharex=True, gridspec_kw={'height_ratios': [1, 7]*len(farmnodes)})
    jobs_counts = {}
    shared_jobs = None
    for id, farmnode in enumerate(farmnodes):
        df_slurm_segs_farmnode= df_slurm_segs[df_slurm_segs['node_name']==farmnode]
        jobs = df_slurm_segs_farmnode['allocation_id'].unique()
        for job in jobs:
            if job not in jobs_counts:
                jobs_counts[job]=1
            else:
                jobs_counts[job]+=1
    shared_jobs = [key for key, value in jobs_counts.items() if value > 1]
    cmap = mpl.colormaps['tab10'].colors
    cmap = tuple_minus(cmap, tuple([(0.4980392156862745, 0.4980392156862745, 0.4980392156862745)]))
    n_cmap = len(cmap)
    cmap2 = tuple_minus(mpl.colormaps['tab20'].colors, mpl.colormaps['tab10'].colors)
    cmap2 = tuple_minus(cmap2, tuple([(0.7803921568627451, 0.7803921568627451, 0.7803921568627451)]))
    n_cmap2 = len(cmap2)
    colors = {}
    
    major_jobs = list(df_slurm_segs_allnode[df_slurm_segs_allnode['duration']>3600].allocation_id.unique())
    all_jobs = list(df_slurm_segs_allnode.allocation_id.unique())
    i = 0
    j = 0
    for job in all_jobs:
        if job in major_jobs:
            if job in shared_jobs:
                colors[job]= cmap[i%n_cmap]
                i += 1
            colors[job]= cmap2[j%n_cmap2]
            j += 1
        else:
            colors[job]= (1.0, 1.0, 1.0)
    
    for id, farmnode in enumerate(farmnodes):
        df_sim_farmnode= df_sim[df_sim['farmnode']==farmnode]
        df_slurm_segs_farmnode= df_slurm_segs[df_slurm_segs['node_name']==farmnode]
        df_sim_farmnode = df_sim_farmnode.reset_index()
        single_line_plot(ax[2*id+1], df_sim_farmnode, 'nei_similarity',x_col='index',zoom=False,color1 = "#006BA4")
        min_tick = 0.5 if df_sim_farmnode['nei_similarity'].min()>=0.5 else 0
        ax[2*id+1].set_yticks(np.linspace(min_tick, 1, 3))
        ax[2*id+1].set_ylabel('Graph Neighbor\n Similarity',fontsize=12, color='k')
        ax[2*id+1].tick_params(axis="y", labelcolor="k",labelsize=10)
        ax[2*id+1].set_xlabel('',fontsize=12, color='k')
        ax[2*id+1].tick_params(axis="x", labelcolor="k",labelsize=10)
        ax[2*id+1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
        
        oldrow = None
        idle_duration = 0
        for index, row in df_slurm_segs_farmnode.iterrows():
            if oldrow is not None:
                idle_duration= int((pd.to_datetime(row['begin_time']) - pd.to_datetime(oldrow['end_time'])).total_seconds())
            else:
                idle_duration= int((pd.to_datetime(row['begin_time']) - pd.to_datetime('2020-01-01 00:00:00')).total_seconds())
            if row.duration>=3600:
                if row.allocation_id in shared_jobs: 
                    rect = patches.Rectangle((row['begin_time'], 0), row['end_time'] - row['begin_time'], 1, facecolor=colors[row.allocation_id], hatch='xx',alpha=0.99)
                else:
                    rect = patches.Rectangle((row['begin_time'], 0), row['end_time'] - row['begin_time'], 1, facecolor=colors[row.allocation_id])
                ax[2*id].add_patch(rect)
                ax[2*id].set_yticks([])
                dates = pd.date_range(start=start_time, end = '2020-01-04 00:00:00', freq='6H')  
                elapsed_hours = ((dates - pd.to_datetime(start_time)).total_seconds() / 3600).astype(int)
                ax[2*id].set_xticks(dates)
                ax[2*id].set_xticklabels(elapsed_hours)

                if row.duration>3600:
                    ax[2*id+1].axvline(x=row['begin_time'], color='grey', linestyle='--', alpha=0.3)
                    ax[2*id+1].axvline(x=row['end_time'], color='grey', linestyle='--', alpha=0.3)


            oldrow = row

    positions = [ax0.get_position() for ax0 in ax]
    for id, farmnode in enumerate(farmnodes):
        i=id*2
        positions[i] = [positions[i].x0, positions[i+1].y0+positions[i+1].height, positions[i].width, positions[i].height]

    for ax0, pos in zip(ax, positions):
        ax0.set_position(pos)
    
    plt.xlim(pd.to_datetime(start_time),pd.to_datetime('2020-01-04 00:00:00')) 

    marker_handles = [
        patches.Patch(facecolor='#c4cad2', edgecolor='black', label='Job running on one compute node'),
        patches.Patch(facecolor='#c4cad2', edgecolor='black', hatch='xx', label='Job running on multiple compute nodes'),
        patches.Patch(facecolor='white', edgecolor='black', label='Idle'),
    ]
    legend1 = plt.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, -0.6),prop={'size': 15}, ncol=4)
    plt.gca().add_artist(legend1)

    plt.savefig(savepath)
    
def organize_slurm(df_slurm,farmnode):
    df_slurm = df_slurm[(df_slurm['node_name']==farmnode)]
    df_slurm = reshape_df_duration(df_slurm)
    df_slurm['begin_time_idx'] = df_slurm['begin_time']
    df_slurm = df_slurm.resample('1h', on='begin_time_idx').apply(custom_resampler)
    df_slurm['sum_duration'] = df_slurm['sum_duration'].apply(lambda x: x.total_seconds())
    df_slurm = df_slurm.reset_index().drop('level_1', axis=1)
    df_slurm.set_index('begin_time_idx', inplace=True)
    time_range_df = pd.date_range(start=start_time, end=end_time, freq='H',inclusive='left')
    df_slurm = df_slurm.reindex(time_range_df, fill_value=np.nan)
    df_slurm['num_jobs'].fillna(0, inplace=True)
    df_slurm['all_jobs'].fillna('{}',inplace=True)
    df_slurm['flag_single_job'].fillna(-1, inplace=True)
    return df_slurm

common_jobs = [789985, 789987, 789994, 790066, 791870, 792117, 793079, 793266, 793274]
farmnodes = ['g10n05','g12n05','g21n05']



start_time = '2020-01-01 00:00:00'
end_time = '2020-01-03 21:00:00'

df_slurm = pd.read_csv(r'./data/olcfcutsec/sanitized_pernode_jobs_202001.csv', engine='pyarrow')
df_slurm['begin_time'] = pd.to_datetime(df_slurm['begin_time'])
df_slurm['end_time'] = pd.to_datetime(df_slurm['end_time'])
df_slurm = slice_df_by_time_range(df_slurm,[[start_time,end_time]],timestamp_col='begin_time')
df_slurm = df_slurm.sort_values(by='begin_time')
df_slurm['end_time'] = df_slurm['end_time'].apply(lambda x: x if x<pd.to_datetime(end_time) else pd.to_datetime(end_time))
df_slurm_original = df_slurm.copy()
df_slurm_original['duration'] = (pd.to_datetime(df_slurm_original['end_time']) - pd.to_datetime(df_slurm_original['begin_time'])).dt.total_seconds().astype(int)

global_g_path ='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/24hour/profile_graph_adj.csv'
global_g_std_path ='results/olcfcutsec_fm_05-14--00-59-47/graph_datasize/24hour/profile_graph_adj_std.csv'
df_global = pd.read_csv(global_g_path)
df_global = pd.DataFrame(convert_adj2edges(df_global),columns=['source','destination','value'])
df_global['source'] = df_global['source'].apply(lambda x: int(x))
df_global['destination'] = df_global['destination'].apply(lambda x: int(x))
df_global_std = pd.read_csv(global_g_std_path)
df_global_std = pd.DataFrame(convert_adj2edges(df_global_std),columns=['source','destination','value'])
df_global_std['source'] = df_global_std['source'].apply(lambda x: int(x))
df_global_std['destination'] = df_global_std['destination'].apply(lambda x: int(x))


graph_file_path2 = "results/olcfcutsec_fm_05-14--00-59-47/fm/graph_evolve_sim_eval/similarity.csv"
df_sim = pd.read_csv(graph_file_path2)

df_sim.rename(columns={'similarity':'nei_similarity'},inplace=True)
df_sim['timestamp'] = pd.to_datetime(df_sim['timestamp'])
df_sim.rename(columns={'timestamp':'index'},inplace=True)
df_sim = df_sim.set_index('index')
farmnodes_str = '_'.join(str(x) for x in farmnodes)
plot_line2(df_slurm_original, df_sim,farmnodes=farmnodes,savepath="./fig8.png")
