import numpy as np
from MaskGNN_interpretation import build_data
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from MaskGNN_interpretation.maskgnn import collate_molgraphs, EarlyStopping, run_a_train_epoch, \
    run_an_eval_epoch, set_random_seed, RGCN, pos_weight
import pickle as pkl
import time


# fix parameters of model
def SMEG_explain_for_substructure(seed, task_name, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128,
               lr=0.0003, classification=True, sub_type='fg'):
    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 128
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '../data/graph_data/{}_for_{}.bin'.format(args['data_name'], sub_type)
    args['group_path'] = '../data/graph_data/{}_group_for_{}.csv'.format(args['data_name'], sub_type)
    args['smask_path'] = '../data/graph_data/{}_smask_for_{}.npy'.format(args['data_name'], sub_type)
    args['seed'] = seed
    
    print('***************************************************************************************************')
    print('{}, seed {}, substructure type {}'.format(args['task_name'], args['seed']+1, sub_type))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = build_data.load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        smask_path=args['smask_path'],
        classification=args['classification'],
        random_shuffle=False
    )
    print("Molecule graph is loaded!")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)
    
    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)
    
    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    if args['classification']:
        pos_weight_np = pos_weight(train_set)
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                    pos_weight=pos_weight_np.to(args['device']))
    else:
        loss_criterion = torch.nn.MSELoss(reduction='none')
    model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                 ffn_dropout=args['ffn_drop_out'],
                 rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                 rgcn_drop_out=args['rgcn_drop_out'],
                 classification=args['classification'])
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'] + '_' + str(seed + 1),
                            mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    pred_name = '{}_{}_{}'.format(args['task_name'], sub_type, seed + 1)
    stop_test_list, _ = run_an_eval_epoch(args, model, test_loader, loss_criterion,
                                          out_path='../prediction/{}/{}_test'.format(sub_type, pred_name))
    stop_train_list, _ = run_an_eval_epoch(args, model, train_loader, loss_criterion,
                                           out_path='../prediction/{}/{}_train'.format(sub_type, pred_name))
    stop_val_list, _ = run_an_eval_epoch(args, model, val_loader, loss_criterion,
                                         out_path='../prediction/{}/{}_val'.format(sub_type, pred_name))
    print('Mask prediction is generated!')



# for task in ['ESOL', 'Mutagenicity', 'hERG']:
for task in ['Mutagenicity']:
    # for sub_type in ['fg', 'murcko', 'brics', 'brics_emerge', 'murcko_emerge']:
    for sub_type in ['fg']:
        # load
        with open('../result/hyperparameter_{}.pkl'.format(task), 'rb') as f:
            hyperparameter = pkl.load(f)
        for i in range(10):
            SMEG_explain_for_substructure(seed=i, task_name=task,
                                          rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                          ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                          lr=hyperparameter['lr'], classification=hyperparameter['classification'], sub_type=sub_type)













