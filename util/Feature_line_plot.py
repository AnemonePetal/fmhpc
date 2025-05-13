import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from util.argparser import get_args
from util.env import prepare_env
from util.Data import Dataset
from sklearn.preprocessing import MinMaxScaler
import json5
from datetime import datetime, timedelta
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42	
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import re
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import textwrap


def wrap_ticks_labels(ax, width, axis="x", break_long_words=False):
    labels = []
    if axis == "x":
        for label in ax.get_xticklabels():
            text = label.get_text()
            labels.append(
                textwrap.fill(text, width=width, break_long_words=break_long_words)
            )
        ax.set_xticklabels(labels, rotation=0)
    elif axis == "y":
        for label in ax.get_yticklabels():
            text = label.get_text()
            labels.append(
                textwrap.fill(text, width=width, break_long_words=break_long_words)
            )
        ax.set_yticklabels(labels, rotation=0)
    else:
        raise ValueError("axis must be x or y")


def rename_feat(feature):
    def replace_last_space_with_newline(s):
        parts = s.rsplit(' ', 1)
        return '\n'.join(parts)
    def replace_last_char_with_newline(s):
        parts = s.rsplit('_', 1)
        return '_\n'.join(parts)

    if 'temp' in feature or 'power' in feature:
        return feature
    if 'cumsum' in feature:
        return 'cumsum'
    feature = feature.replace("node_memory_", "mem:\n")
    feature = feature.replace("node_disk_", "disk:\n")
    feature = re.sub(r"^(?!mem:|disk:)", "cpu:\n", feature)
    feature = feature.replace("(cumsum)", "(cumsum)")
    feature = feature.replace("(diff)", "(diff)")
    feature = feature.replace("_", " ")
    if feature == 'mem:\nWriteback bytes': feature = 'mem: Writeback bytes'
    if feature == 'disk:\nread time seconds total': feature = 'disk: read time seconds total'
    if feature == 'cpu:\nirq': feature = 'cpu: irq'
    return feature


def load_old_args(args):
    with open(os.path.dirname(args.paths["test_re"]) + "/args.json") as f:
        data_args = json5.load(f)
        for key in data_args:
            setattr(args, key, data_args[key])
    return 0


def get_mask_by_time_range(df, time_ranges, farmnodes=None):
    for start_time, end_time in time_ranges:
        if start_time > end_time:
            raise ValueError("Start time must be less than end time")
        if start_time == "":
            mask = df["timestamp"] <= end_time
        elif end_time == "":
            mask = df["timestamp"] >= start_time
        else:
            mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        if farmnodes != None:
            if isinstance(farmnodes, list):
                mask = mask & (df["instance"].isin(farmnodes))
            elif isinstance(farmnodes, str):
                mask = mask & (df["instance"] == farmnodes)
            else:
                raise ValueError("farmnodes must be a list or a string")
    return mask


def add_stat_data(data, features, stat):
    features_stat = [feature + "(" + stat + ")" for feature in features]
    if stat == "cumsum":
        data[features_stat] = data.groupby("instance")[features].transform("cumsum")
    elif stat == "diff":
        data[features_stat] = data.groupby("instance")[features].transform(
            lambda x: x.diff()
        )


def normalize_instance(data, features_stat):
    scaler = MinMaxScaler()
    data[features_stat] = scaler.fit_transform(data[features_stat])
    return data


def add_extra_features(args, features, stat):
    for feature in features:
        args.features.append(feature + "(" + stat + ")")
    return args


def retrieve_data(args, fpath_groups=[[1, 58, 35], [30, 54]]):
    args.no_store = True
    prepare_env(args)
    args.slide_stride = 1
    args.scaler_only_fit = True
    if args.attack:
        args.retain_beforeattack = True
    data = Dataset(args)

    fpath_feats = [[args.features[i] for i in g] for g in fpath_groups]
    columns = data.train.columns
    fpath_groups_flat = list(
        set([item for sublist in fpath_groups for item in sublist])
    )
    features = [args.features[i] for i in fpath_groups_flat]
    non_features_columns = [
        col for col in data.train.columns if col not in args.features
    ]
    old_features = features.copy()
    data.train = data.train[non_features_columns + features]

    add_stat_data(data.train, features, "cumsum")

    add_stat_data(data.train, features, "diff")
    args.features = features
    old_features = features.copy()
    add_extra_features(args, old_features, "cumsum")
    add_extra_features(args, old_features, "diff")

    data.train = data.train.groupby("instance", group_keys=False).apply(
        normalize_instance,
        features_stat=[f for f in args.features if f not in old_features],
    )
    return data, fpath_feats


def plus_min(timestamp, min=1):
    time_format = "%Y-%m-%d %H:%M:%S"
    if min > 0:
        dt = datetime.strptime(timestamp, time_format)
        next_min = dt + timedelta(minutes=min)
    else:
        min = -min
        dt = datetime.strptime(timestamp, time_format)
        next_min = dt - timedelta(minutes=min)
    return next_min.strftime(time_format)


def twin_line_plot(ax1, df, feats, x_col='timestamp',time_range=None, zoom=True, float_y_flag = False):
    color1 = "k"
    ax_max = df[feats[0]].max()
    if ax_max > 100:
        exponent_axis = np.floor(np.log10(ax_max)).astype(int)
        df[feats[0]] = df[feats[0]] / 10**exponent_axis

    sns.lineplot(data=df, x=x_col, y=feats[0], ax=ax1, alpha=0.5,color="k")
    hours = mdates.HourLocator(interval=1)
    h_fmt = mdates.DateFormatter("%H")

    if not float_y_flag:
        ax1.set_ylabel(
            rename_feat(feats[0]),
            color=color1,
            fontsize=23,
        )
    else:
        ax1.set_ylabel(
            '',
            color=color1,
            fontsize=0,
        )
        ax1.annotate(
            rename_feat(feats[0]),
            color=color1,
            fontsize=19,
            xy=(0.02, 0.03),
            xycoords="axes fraction",
        )


    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_yticks(np.linspace(df[feats[0]].min(), df[feats[0]].max(), 3))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1.yaxis.set_tick_params(labelsize=22)

    color2 = 'b'
    ax2 = ax1.twinx()
    norm_df = df
    sns.lineplot(
        data=norm_df,
        x=x_col,
        y=feats[1],
        ax=ax2,
        color=color2,
        linestyle=(0, (5, 1)),
        linewidth=2,
    )
    
    if not float_y_flag:
        ax2.set_ylabel(
            rename_feat(feats[1]),
            color=color2,
            fontsize=23,
        )
    else:
        ax2.set_ylabel(
            '',
            color=color1,
            fontsize=0,
        )
        ax2.annotate(
            rename_feat(feats[1]),
            color=color2,
            fontsize=19,
            xy=(0.73, 0.93),
            xycoords="axes fraction",
        )


    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yticks(np.linspace(norm_df[feats[1]].min(), norm_df[feats[1]].max(), 3))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax2.yaxis.set_tick_params(labelsize=22)
    
    if zoom == True:
        zoom_in_timerange = [["2023-05-19 14:00:00", "2023-05-19 15:00:00"]]
        axins = inset_axes(ax1, width=2, height=0.2 * 2, loc="upper left")
        mask = get_mask_by_time_range(df, zoom_in_timerange)
        df_zoom = df[mask]
        sns.lineplot(data=df, x=x_col, y=feats[0], ax=axins, color=color1)


        start_time = datetime.strptime(zoom_in_timerange[0][0], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(zoom_in_timerange[0][1], "%Y-%m-%d %H:%M:%S")
        axins.set_xlim([start_time, end_time])
        axins.set_xlim([start_time, end_time])
        df_zoom_rang = df_zoom[feats[0]].max() - df_zoom[feats[0]].min()
        if df_zoom_rang == 0:
            axins.set_ylim([df_zoom[feats[0]].min() - 0.05, df_zoom[feats[0]].max() + 0.05])
        else:
            axins.set_ylim(
                [
                    df_zoom[feats[0]].min() - 0.15 * df_zoom_rang,
                    df_zoom[feats[0]].max() + 0.15 * df_zoom_rang,
                ]
            )

        axins.set_xticks(
            np.linspace(axins.get_xticks().min(), axins.get_xticks().max(), 5)
        )
        axins.set_xticklabels(range(5))
        axins.set_xlabel("")
        axins.yaxis.set_visible(False)

        pp, p1, p2 = mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5")

        ax1.yaxis.tick_left()
        ax2.yaxis.tick_right()

        if ax_max > 100:
            y_min, y_max = ax1.get_ylim()
            ticks = [(tick - y_min) / (y_max - y_min) for tick in ax1.get_yticks()]

            ax1.annotate(
                r"$\times$10$^{%i}$" % (exponent_axis),
                xy=(0.01, ticks[-1] * 0.93),
                xycoords="axes fraction",
                color=color1,
            )
    return ax2

def single_line_plot(ax1, df, feat, x_col='timestamp', time_range=None, zoom=False, zoom_position='upper left',color1 = "k",float_y_flag=False,xy=(-1,-1)):
    ax_max = df[feat].max()
    if ax_max > 100:
        exponent_axis = np.floor(np.log10(ax_max)).astype(int)
        df[feat] = df[feat] / 10**exponent_axis

    sns.lineplot(data=df, x=x_col, y=feat, ax=ax1, color=color1,linewidth=2)
    if not float_y_flag:
        ax1.set_ylabel(
            rename_feat(feat),
            color=color1,
            fontsize=23,
        )
    else:
        ax1.set_ylabel(
            '',
            color=color1,
            fontsize=0,
        )
        if xy==(-1,-1):
            xy = (0.02, 0.03)

        ax1.annotate(
            rename_feat(feat),
            color=color1,
            fontsize=19,
            xy=xy,
            xycoords="axes fraction",
        )

    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax1.yaxis.set_tick_params(labelsize=22)

    if zoom == True:
        zoom_in_timerange = [["2023-05-19 14:00:00", "2023-05-19 15:00:00"]]
        axins = inset_axes(ax1, width=2, height=0.2*2, loc= zoom_position)
        mask = get_mask_by_time_range(df, zoom_in_timerange)
        df_zoom = df[mask]
        sns.lineplot(data=df, x=x_col, y=feat, ax=axins, color=color1)

        start_time = datetime.strptime(zoom_in_timerange[0][0], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(zoom_in_timerange[0][1], "%Y-%m-%d %H:%M:%S")
        axins.set_xlim([start_time, end_time])
        df_zoom_rang = df_zoom[feat].max() - df_zoom[feat].min()
        if df_zoom_rang == 0:
            axins.set_ylim([df_zoom[feat].min() - 0.05, df_zoom[feat].max() + 0.05])
        elif df_zoom[feat].min() == 0:
            axins.set_ylim([-0.1, df_zoom[feat].max() + 0.15 * df_zoom_rang])
        else:
            axins.set_ylim(
                [
                    df_zoom[feat].min() - 0.15 * df_zoom_rang,
                    df_zoom[feat].max() + 0.15 * df_zoom_rang,
                ]
            )
        if zoom_position == 'upper left':
            pp,p1,p2 = mark_inset(ax1, axins, loc1=3, loc2=4, fc="none", ec="0.5")
        elif zoom_position == 'lower left':
            pp, p1, p2 = mark_inset(ax1, axins, loc1=1, loc2=2, fc="none", ec="0.5")
        axins.yaxis.set_visible(False)
        axins.set_xticks(
            np.linspace(axins.get_xticks().min(), axins.get_xticks().max(), 5)
        )
        axins.set_xticklabels(range(5))
        axins.set_xlabel("")
        if zoom_position == 'lower left':
            axins.xaxis.tick_top()


    if ax_max > 100:
        y_min, y_max = ax1.get_ylim()
        ticks = [(tick - y_min) / (y_max - y_min) for tick in ax1.get_yticks()]
        if df[feat].min() < df[feat].max():
            ax1.annotate(
                r"$\times$10$^{%i}$" % (exponent_axis),
                xy=(0.01, 0.85),
                xycoords="axes fraction",
                color=color1,
                fontsize=22,
            )
        else:
            ax1.annotate(
                r"$\times$10$^{%i}$" % (exponent_axis),
                xy=(0.01, 0.93),
                xycoords="axes fraction",
                color=color1,
                fontsize=22,
            )

def line_plot_wrapper(df, feats, time_range, savefile, farmnode="farm140109:9100",y_min_1=-1,y_max_1=-1,title=""):
    mask = get_mask_by_time_range(df, time_range, farmnodes=farmnode)
    df = df[mask]
    feats_0 = [feats[0] + suffix for suffix in ["", "(cumsum)"]]
    fig, ax = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.16, right=0.9, top=0.95, bottom=0.05)


    right_ax = twin_line_plot(ax[0], df, feats_0, time_range=time_range, zoom=False,float_y_flag=True)

    y_min_0,y_max_0 = 0,3.2
    ax[0].set_ylim(y_min_0-(y_max_0-y_min_0)*0.15,y_max_0+(y_max_0-y_min_0)*0.1)
    ax[0].set_yticks(np.linspace(y_min_0, y_max_0, 3))
    y_min_0_r,y_max_0_r = 0,1
    right_ax.set_ylim(y_min_0_r-(y_max_0_r-y_min_0_r)*0.15,y_max_0_r+(y_max_0_r-y_min_0_r)*0.1)


    single_line_plot(ax[1], df, feats[1], time_range=time_range,zoom=False,float_y_flag=True)
    ax[1].set_ylim(y_min_1-(y_max_1-y_min_1)*0.15,y_max_1+(y_max_1-y_min_1)*0.1)
    ax[1].set_yticks(np.linspace(y_min_1, y_max_1, 3))


    for a in ax:
        a.xaxis.set_visible(False)

    plt.savefig(savefile)

def line_plot_wrapper2(df, feats, time_range, savefile, farmnode="farm140109:9100",y_min_1=-1,y_max_1=-1,title=""):
    mask = get_mask_by_time_range(df, time_range, farmnodes=farmnode)
    df = df[mask]
    feats_0 = [feats[0] + suffix for suffix in ["", "(cumsum)"]]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(left=0.16, right=0.9, top=0.95, bottom=0.05)



    single_line_plot(ax[0], df, feats[0], time_range=time_range,zoom=False, float_y_flag=True)
    y_min,y_max = 25,55
    ax[0].set_ylim(y_min-(y_max-y_min)*0.1,y_max+(y_max-y_min)*0.1)
    ax[0].set_yticks(np.linspace(y_min, y_max, 3))

    single_line_plot(ax[1], df, feats[1], time_range=time_range,zoom=False,float_y_flag=True)
    ax[1].set_ylim(y_min_1-(y_max_1-y_min_1)*0.15,y_max_1+(y_max_1-y_min_1)*0.1)
    ax[1].set_yticks(np.linspace(y_min_1, y_max_1, 3))


    for a in ax:
        a.xaxis.set_visible(False)

    plt.savefig(savefile)

    
if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from util.argparser import get_args

    args = get_args()
    args.dataset = 'jlab'
    data, fpath_feats = retrieve_data(
        args, fpath_groups=[[54,59],[54,2]]
    )
    time_ranges = [["2023-05-18 10:00:00", "2023-05-24 22:00:00"]]
    line_plot_wrapper(data.train, fpath_feats[0], time_ranges, savefile = 'fig7_c',y_min_1=0.2,y_max_1=1.2)
    line_plot_wrapper(data.train, fpath_feats[1], time_ranges, savefile = 'fig7_d',y_min_1=0,y_max_1=0)
 
 
    args = get_args()
    args.dataset = 'olcfcutsec'
    data, fpath_feats = retrieve_data(
        args, fpath_groups=[[18,5],[18,26]]
    )
    time_ranges = [["2020-01-01 00:00:00", "2020-01-31 00:00:00"]]
    line_plot_wrapper2(data.train, fpath_feats[0], time_ranges, savefile = 'fig7_a', farmnode='a30n04',y_min_1=0,y_max_1=4)
    line_plot_wrapper2(data.train, fpath_feats[1], time_ranges, savefile = 'fig7_b', farmnode='a30n04',y_min_1=0,y_max_1=0)
 




