import torch 
from torch.utils.data import DataLoader
from time import time
from util.TimeDataset import TimeDataset
from util.preprocess import build_loc_net, construct_data
from util.net_prep import get_fc_graph_struc
from util.TimeDataset_2D import TimeDataset2D

def set_dataloader(dataset,args):
    build_timedataset(dataset,args)
    build_loader(dataset,args)

def build_timedataset(dataset,args):
    fc_struc = get_fc_graph_struc(args.paths['dataset'])
    fc_edge_index = torch.tensor(build_loc_net(fc_struc, feature_map=args.features), dtype = torch.long)
    dataset.edge_index_sets=[fc_edge_index]
    if args.model == 'fmnm':
        dataset.train_dataset = TimeDataset2D(dataset.train,args.features, fc_edge_index, mode='train', config=args,timestamp=dataset.train.timestamp)
        if hasattr(args,'test_ignoresync') and args.test_ignoresync == True:
            dataset.test_dataset = TimeDataset2D(dataset.test,args.features, fc_edge_index, mode='test_ignoresync', config=args,timestamp=dataset.test.timestamp)
        else:
            dataset.test_dataset = TimeDataset2D(dataset.test,args.features, fc_edge_index, mode='test', config=args,timestamp=dataset.test.timestamp)
        dataset.val_dataset = TimeDataset2D(dataset.val, args.features, fc_edge_index, mode='train', config=args,timestamp=dataset.val.timestamp)
        if hasattr(args, 'retain_beforeattack') and args.retain_beforeattack and hasattr(dataset, 'test_before_attack'):
            dataset.test_before_attack_dataset = TimeDataset2D(dataset.test_before_attack,args.features, fc_edge_index, mode='test', config=args,timestamp=dataset.test_before_attack.timestamp)
    else:
        dataset.train_dataset = TimeDataset(dataset.train,args.features, fc_edge_index, mode='train', config=args,timestamp=dataset.train.timestamp)
        if hasattr(args,'test_ignoresync') and args.test_ignoresync == True:
            dataset.test_dataset = TimeDataset(dataset.test,args.features, fc_edge_index, mode='test_ignoresync', config=args,timestamp=dataset.test.timestamp)
        else:
            dataset.test_dataset = TimeDataset(dataset.test,args.features, fc_edge_index, mode='test', config=args,timestamp=dataset.test.timestamp)
        dataset.val_dataset = TimeDataset(dataset.val, args.features, fc_edge_index, mode='train', config=args,timestamp=dataset.val.timestamp)
        if hasattr(args, 'retain_beforeattack') and args.retain_beforeattack and hasattr(dataset, 'test_before_attack'):
            dataset.test_before_attack_dataset = TimeDataset(dataset.test_before_attack,args.features, fc_edge_index, mode='test', config=args,timestamp=dataset.test_before_attack.timestamp)

def build_loader(dataset,args):
    if args.dataloader_num_workers >0:
        dataset.train_dataloader = DataLoader(dataset.train_dataset, batch_size=args.batch, shuffle=True,num_workers= args.dataloader_num_workers,pin_memory=True,prefetch_factor=args.dataloader_prefetch_factor,persistent_workers=True)
        dataset.val_dataloader = DataLoader(dataset.val_dataset, batch_size=args.batch, shuffle=False,num_workers= args.dataloader_num_workers,pin_memory=True,prefetch_factor=args.dataloader_prefetch_factor,persistent_workers=True)
        dataset.test_dataloader = DataLoader(dataset.test_dataset, batch_size=args.batch,shuffle=False,num_workers= args.dataloader_num_workers,pin_memory=True,prefetch_factor=args.dataloader_prefetch_factor,persistent_workers=True)
    else:
        dataset.train_dataloader = DataLoader(dataset.train_dataset, batch_size=args.batch, shuffle=False,pin_memory=True)
        dataset.val_dataloader = DataLoader(dataset.val_dataset, batch_size=args.batch, shuffle=False,pin_memory=True)
        dataset.test_dataloader = DataLoader(dataset.test_dataset, batch_size=args.batch,shuffle=False,pin_memory=True)

    if hasattr(args, 'retain_beforeattack') and args.retain_beforeattack:
        dataset.test_before_attack_dataloader = DataLoader(dataset.test_before_attack_dataset, batch_size=args.batch,shuffle=False,num_workers= 0,pin_memory=True)


def test_loader_speed(dataset,args):
    for num_prefetch in range(4, 10, 2):
        for num_workers in range(2, 10, 2):  
            train_loader = DataLoader(dataset.train_dataset,batch_size=args.batch, shuffle=False,num_workers=num_workers,pin_memory=True,prefetch_factor=num_prefetch)
            start = time()
            for epoch in range(1, 3):
                for i, data in enumerate(train_loader, 0):
                    pass
            end = time()
            print("Finish with:{} second, num_workers={}, num_prefetch={}".format((end - start)/2, num_workers, num_prefetch))

def set_miniloader(df,args):
    fc_struc = get_fc_graph_struc(args.paths['dataset'])
    fc_edge_index = torch.tensor(build_loc_net(fc_struc, feature_map=args.features), dtype = torch.long)
    timedataset = TimeDataset(df, args.features, fc_edge_index, mode='train', config=args,timestamp=df.timestamp)
    return DataLoader(timedataset, batch_size=args.batch, shuffle=False,num_workers= 0,pin_memory=True)
