from util.Data import filter_df

def check_consisitency(df_re,df_gt,reset_index=True):
    if reset_index:
        df_re = df_re.reset_index(drop=True)
        df_gt = df_gt.reset_index(drop=True)

    if (df_re['timestamp']== df_gt['timestamp']).all() == False or df_re.shape[0]!=df_gt.shape[0]:
        raise ValueError('timestamp not match')
    if (df_re['instance'] == df_gt['instance']).all() == False:
        raise ValueError('instance not match')


def align_df(re,gt,rang,args ,data_filter={}):
    rang = rang[:,-args.pred_win:].flatten()
    gt = gt.iloc[rang]
    if data_filter!={}:
        re=filter_df(re, data_filter)
        gt=filter_df(gt, data_filter)
    check_consisitency(re,gt)
    re = re.reset_index(drop=True)
    gt = gt.reset_index(drop=True)
    return gt

def sync_sort(df1,df2,by_column):
    assert (df1.index == df2.index).all()
    df1 = df1.sort_values(by_column)
    df2 = df2.sort_values(by_column)
    check_consisitency(df1,df2)
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    return df1,df2

def align_df_2d(re,gt,rang,rang_hnode,args,data_filter={}):
    rang_hnode = rang_hnode.flatten()
    rang = rang[rang_hnode]
    rang = rang[:,-args.pred_win:].flatten()
    gt = gt.iloc[rang]
    if data_filter!={}:
        re=filter_df(re, data_filter)
        gt=filter_df(gt, data_filter)
    check_consisitency(re,gt)
    re = re.reset_index(drop=True)
    gt = gt.reset_index(drop=True)
    return gt
